"""
Training Loop for DTLN (Waveform Model).
PS 26052 -- Adaptive Defence ANC.

DTLN is a waveform-in / waveform-out model with internal STFT/iSTFT.
It cannot use the spectral-mask training loop in train.py. This module
mirrors the same data loading, seed, provenance, and PH1 input_mode
contract but feeds waveforms directly to DTLNNet.forward().

Loss: SI-SDR (scale-invariant signal-to-distortion ratio).
"""

import csv
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.dsp.vss_nlms import VSSNLMSFilter
from src.ai.train import DatasetLoader, _sha256_of_file, _prepare_residual_inputs


def si_sdr_loss(estimated: "torch.Tensor", reference: "torch.Tensor") -> "torch.Tensor":
    """Negative SI-SDR loss (for minimization)."""
    ref = reference - torch.mean(reference)
    est = estimated - torch.mean(estimated)
    dot = torch.sum(est * ref)
    s_target = dot / (torch.sum(ref ** 2) + 1e-8) * ref
    e_noise = est - s_target
    si_sdr = 10 * torch.log10(torch.sum(s_target ** 2) / (torch.sum(e_noise ** 2) + 1e-8) + 1e-8)
    return -si_sdr


def train_dtln_model(
    model: "nn.Module",
    metadata_csv: str = "data/v4/metadata/metadata_v4.csv",
    epochs: int = 30,
    lr: float = 1e-4,
    device: str = "cpu",
    save_path: Optional[str] = None,
    train_split: str = "TRAIN",
    val_split: str = "TEST_A_UNSEEN_SPEAKER",
    seed: int = 42,
    input_mode: str = "RAW_PRIMARY",
) -> Dict[str, Any]:
    """
    Train DTLN on DATASET_V004.

    DTLN forward: (B, 1, T) -> (B, 1, T).
    Loss: SI-SDR only (waveform model, no spectral L1).

    Args:
        model: DTLNNet instance.
        metadata_csv: Path to dataset metadata.
        epochs: Number of training epochs.
        lr: Learning rate.
        device: 'cpu' or 'cuda'.
        save_path: Checkpoint save path.
        train_split: Split for training.
        val_split: Split for validation.
        seed: Random seed.
        input_mode: 'RAW_PRIMARY' or 'NLMS_RESIDUAL'.

    Returns:
        Training history dict with checkpoint_sha256 if saved.
    """
    if not TORCH_AVAILABLE:
        return {"status": "SKIPPED", "reason": "PyTorch not available"}
    if input_mode not in ("RAW_PRIMARY", "NLMS_RESIDUAL"):
        raise ValueError(f"input_mode must be 'RAW_PRIMARY' or 'NLMS_RESIDUAL', got {input_mode!r}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    meta_path = Path(metadata_csv)
    data_root = meta_path.parent.parent

    train_loader = DatasetLoader(str(meta_path), str(data_root), split_filter=train_split)
    val_loader = DatasetLoader(str(meta_path), str(data_root), split_filter=val_split)
    train_loader.assert_nonempty("train")
    val_loader.assert_nonempty("validation")

    overlap = train_loader.speaker_ids() & val_loader.speaker_ids()
    if overlap:
        raise RuntimeError(f"Train/val speaker overlap: {overlap}")

    # Precompute NLMS residuals if needed
    train_inputs = None
    val_inputs = None
    if input_mode == "NLMS_RESIDUAL":
        print("[DTLN-TRAIN] Precomputing VSS-NLMS residuals (train set)...", flush=True)
        train_inputs = _prepare_residual_inputs(train_loader)
        print("[DTLN-TRAIN] Precomputing VSS-NLMS residuals (val set)...", flush=True)
        val_inputs = _prepare_residual_inputs(val_loader)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "val_loss": [], "epoch_time_s": []}
    param_count = sum(p.numel() for p in model.parameters())

    print(f"\n[DTLN-TRAIN] Model: {model.__class__.__name__}, Params: {param_count:,d}", flush=True)
    print(f"[DTLN-TRAIN] Epochs: {epochs}, LR: {lr}, Device: {device}, Seed: {seed}", flush=True)
    print(f"[DTLN-TRAIN] Input: {input_mode}", flush=True)
    print(f"[DTLN-TRAIN] Loss: SI-SDR (waveform domain)", flush=True)
    print(f"[DTLN-TRAIN] Train: {len(train_loader)} clips, Val: {len(val_loader)} clips", flush=True)

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        epoch_loss = 0.0

        for i in range(len(train_loader)):
            clean, primary, reference, _ = train_loader[i]
            clean_t = torch.from_numpy(clean).float().to(device)
            noisy_wav = train_inputs[i] if train_inputs is not None else primary
            noisy_t = torch.from_numpy(noisy_wav).float().to(device)

            # DTLN expects (B, 1, T)
            noisy_in = noisy_t.unsqueeze(0).unsqueeze(0)
            enhanced = model(noisy_in).squeeze(0).squeeze(0)  # (T,)

            # Align lengths
            min_len = min(len(enhanced), len(clean_t))
            loss = si_sdr_loss(enhanced[:min_len], clean_t[:min_len])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_train = epoch_loss / max(len(train_loader), 1)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i in range(len(val_loader)):
                clean, primary, _, _ = val_loader[i]
                clean_t = torch.from_numpy(clean).float().to(device)
                noisy_wav = val_inputs[i] if val_inputs is not None else primary
                noisy_t = torch.from_numpy(noisy_wav).float().to(device)

                noisy_in = noisy_t.unsqueeze(0).unsqueeze(0)
                enhanced = model(noisy_in).squeeze(0).squeeze(0)
                min_len = min(len(enhanced), len(clean_t))
                v_loss = si_sdr_loss(enhanced[:min_len], clean_t[:min_len])
                val_loss += v_loss.item()

        avg_val = val_loss / max(len(val_loader), 1)
        epoch_time = time.perf_counter() - t0

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["epoch_time_s"].append(round(epoch_time, 2))

        print(f"  Epoch {epoch:03d}/{epochs:03d} | Train: {avg_train:.4f} | "
              f"Val: {avg_val:.4f} | Time: {epoch_time:.1f}s", flush=True)

    result = {
        "status": "COMPLETED",
        "model": model.__class__.__name__,
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "total_time_s": sum(history["epoch_time_s"]),
        "history": history,
    }

    if save_path:
        save_p = Path(save_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), str(save_p))
        checkpoint_hash = _sha256_of_file(save_p)
        result["checkpoint_path"] = str(save_p.resolve())
        result["checkpoint_sha256"] = checkpoint_hash
        print(f"\n[+] Checkpoint saved: {save_p.resolve()}", flush=True)
        print(f"[+] Checkpoint SHA-256: {checkpoint_hash}", flush=True)

        run_metadata = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model_class": model.__class__.__name__,
            "model_parameter_count": param_count,
            "checkpoint_path": str(save_p.resolve()),
            "checkpoint_sha256": checkpoint_hash,
            "seed": seed,
            "epochs": epochs,
            "lr": lr,
            "device": device,
            "metadata_csv": str(meta_path.resolve()),
            "train_split": train_split,
            "train_samples": len(train_loader),
            "val_split": val_split,
            "val_samples": len(val_loader),
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss": history["val_loss"][-1],
            "torch_version": torch.__version__,
            "training_input": input_mode,
            "loss_function": "SI-SDR",
        }
        sidecar_path = save_p.with_suffix(".json")
        with open(sidecar_path, "w", encoding="utf-8") as f:
            json.dump(run_metadata, f, indent=2)
        print(f"[+] Run metadata saved: {sidecar_path.resolve()}", flush=True)
        result["run_metadata_path"] = str(sidecar_path.resolve())

    return result
