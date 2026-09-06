"""
Training Loop for Speech Enhancement Neural Networks.
PS 26052 -- Adaptive Defence ANC.

Supports training any model that follows the spectral mask interface:
  - TinyEnhancer V3 (Conv2D mask estimator)
  - DTLN (Dual-signal Transformation LSTM)
  - CRN-Micro (Convolutional Recurrent Network)

Loss functions:
  - L1 loss on magnitude spectrogram
  - SI-SDR loss (scale-invariant SDR)
  - Combined: alpha * L1_mag + (1-alpha) * (-SI-SDR)

Training data: DATASET_V004 (data/v4/)

PH0.7 CHANGES FROM ORIGINAL
----------------------------
1. FIXED: val split filter was "test_a" (exact-match) but the CSV's actual
   split value is "TEST_A_UNSEEN_SPEAKER" -- exact match silently returned
   0 rows, so validation ran on an empty set every epoch (Val printed as
   0.0000, which reads as "suspiciously good" rather than "broken").
2. ADDED: hard failure if train or val split resolves to 0 rows, instead
   of only checking the train split.
3. ADDED: explicit disjoint-speaker check between train and val splits,
   since "unseen speaker" is the entire point of that held-out split.
4. ADDED: a fixed random seed (previously unset -- meant re-running
   training could not be verified to reproduce the same result, which
   blocks the PH0.7 Gate 4 reproducibility requirement).
5. ADDED: checkpoint SHA-256 hash + a run-metadata JSON sidecar saved next
   to the checkpoint, so every trained artifact is traceable to the exact
   config, data split, and code state that produced it.
6. FIXED (Gate 2): torch.stft / torch.istft now explicitly use center=False
   to match the project's causal deployment convention. The causality verifier
   treats center=True (PyTorch default) as a causality violation because it
   pads frame_size/2 zeros on each side, introducing future-dependent samples.
7. NOTE: IRM variable is computed but NOT used in the loss function.
   The actual loss is alpha * L1(enhanced_mag, clean_mag) + (1-alpha) * (-SI-SDR).
   IRM exists only as a diagnostic reference -- it is NOT an active
   training target. Do not add IRM loss without explicit design approval.
8. WARNING: Training currently uses RAW PRIMARY MIC as input to TinyEnhancer.
   Deployment feeds NLMS RESIDUAL to TinyEnhancer. This is a known
   train/deploy mismatch documented in PH07_AI_INPUT_CONTRACT.md.
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


def si_sdr_loss(estimated: "torch.Tensor", reference: "torch.Tensor") -> "torch.Tensor":
    """Negative SI-SDR loss (for minimization)."""
    ref = reference - torch.mean(reference)
    est = estimated - torch.mean(estimated)

    dot = torch.sum(est * ref)
    s_target = dot / (torch.sum(ref ** 2) + 1e-8) * ref
    e_noise = est - s_target

    si_sdr = 10 * torch.log10(torch.sum(s_target ** 2) / (torch.sum(e_noise ** 2) + 1e-8) + 1e-8)
    return -si_sdr  # Minimize negative SI-SDR = Maximize SI-SDR


class DatasetLoader:
    """Loads DATASET_V004 samples for training. Exact-match on the 'split' column."""

    def __init__(self, metadata_csv: str, data_root: str, split_filter: str = "TRAIN"):
        self.records = []
        self.split_filter = split_filter
        with open(metadata_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            all_rows = list(reader)
        self._all_split_values = sorted(set(r["split"] for r in all_rows))
        for row in all_rows:
            if row["split"].upper() == split_filter.upper():
                self.records.append(row)
        self.data_root = Path(data_root)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records[idx]
        clean, _ = sf.read(str(self.data_root / row["clean_path"]), dtype="float32")
        primary, _ = sf.read(str(self.data_root / row["primary_path"]), dtype="float32")
        reference, _ = sf.read(str(self.data_root / row["reference_path"]), dtype="float32")
        return clean, primary, reference, row

    def speaker_ids(self):
        return set(r.get("speaker_id") for r in self.records)

    def assert_nonempty(self, role: str):
        if len(self.records) == 0:
            raise RuntimeError(
                f"DatasetLoader for role='{role}' matched 0 rows with "
                f"split_filter='{self.split_filter}'. Actual split values "
                f"present in the CSV are: {self._all_split_values}. "
                f"Fix split_filter to match one of these exactly."
            )


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def train_spectral_mask_model(
    model: "nn.Module",
    metadata_csv: str = "data/v4/metadata/metadata_v4.csv",
    epochs: int = 30,
    lr: float = 1e-3,
    frame_size: int = 256,
    hop_size: int = 128,
    alpha: float = 0.7,
    device: str = "cpu",
    save_path: Optional[str] = None,
    train_split: str = "TRAIN",
    val_split: str = "TEST_A_UNSEEN_SPEAKER",
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Train a spectral mask model on DATASET_V004.

    Args:
        model: PyTorch nn.Module with forward(x) -> mask in [0,1].
        metadata_csv: Path to dataset metadata.
        epochs: Number of training epochs.
        lr: Learning rate.
        frame_size: STFT frame size.
        hop_size: STFT hop size.
        alpha: Weight for L1 loss vs SI-SDR loss.
        device: 'cpu' or 'cuda'.
        save_path: Optional checkpoint save path. If given, a run-metadata
            JSON sidecar is written next to it (same stem, .json suffix).
        train_split: Exact 'split' column value to use for training.
        val_split: Exact 'split' column value to use for validation.
            Defaults to the genuinely disjoint-speaker held-out split.
        seed: Random seed for torch/numpy, set before model init and
            training so runs are reproducible (PH0.7 Gate 4).

    Returns:
        Training history dict, including checkpoint_sha256 if a checkpoint
        was saved.
    """
    if not TORCH_AVAILABLE:
        return {"status": "SKIPPED", "reason": "PyTorch not available"}

    torch.manual_seed(seed)
    np.random.seed(seed)

    meta_path = Path(metadata_csv)
    data_root = meta_path.parent.parent

    # Load training and validation data
    train_loader = DatasetLoader(str(meta_path), str(data_root), split_filter=train_split)
    val_loader = DatasetLoader(str(meta_path), str(data_root), split_filter=val_split)

    # FAIL LOUD instead of silently training/validating on an empty set.
    train_loader.assert_nonempty("train")
    val_loader.assert_nonempty("validation")

    # "Unseen speaker" validation is only meaningful if it's actually disjoint.
    overlap = train_loader.speaker_ids() & val_loader.speaker_ids()
    if overlap:
        raise RuntimeError(
            f"Train and validation splits share speaker_id(s) {overlap} -- "
            f"the '{val_split}' split is supposed to hold out unseen "
            f"speakers. This is a data-leakage condition; fix the split "
            f"before training."
        )

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    window = torch.hann_window(frame_size).to(device)

    history = {"train_loss": [], "val_loss": [], "epoch_time_s": []}

    param_count = sum(p.numel() for p in model.parameters())
    print(f"\n[TRAIN] Model: {model.__class__.__name__}, "
          f"Params: {param_count:,d}", flush=True)
    print(f"[TRAIN] Epochs: {epochs}, LR: {lr}, Device: {device}, Seed: {seed}", flush=True)
    print(f"[TRAIN] STFT: frame={frame_size}, hop={hop_size}, window=hann", flush=True)
    print(f"[TRAIN] WARNING: Training input = RAW PRIMARY MIC (deployment uses NLMS residual)", flush=True)
    print(f"[TRAIN] Loss: alpha={alpha} * L1_mag + {1-alpha} * (-SI-SDR). IRM is NOT in the loss.", flush=True)
    print(f"[TRAIN] Train samples: {len(train_loader)} (split='{train_split}', "
          f"speakers={sorted(train_loader.speaker_ids())})", flush=True)
    print(f"[TRAIN] Val samples: {len(val_loader)} (split='{val_split}', "
          f"speakers={sorted(val_loader.speaker_ids())})", flush=True)

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        epoch_loss = 0.0

        for i in range(len(train_loader)):
            clean, primary, reference, _ = train_loader[i]

            clean_t = torch.from_numpy(clean).float().to(device)
            primary_t = torch.from_numpy(primary).float().to(device)

            # center=False is mandatory for causal consistency (Gate 2).
            clean_stft = torch.stft(clean_t, frame_size, hop_size, window=window, return_complex=True, center=False)
            noisy_stft = torch.stft(primary_t, frame_size, hop_size, window=window, return_complex=True, center=False)

            clean_mag = clean_stft.abs()
            noisy_mag = noisy_stft.abs()
            noisy_phase = noisy_stft.angle()

            # IRM is computed for diagnostic reference ONLY -- it is NOT
            # included in the loss function. See docstring note 7.
            irm = clean_mag / (noisy_mag + 1e-8)
            irm = torch.clamp(irm, 0, 1)

            noisy_input = noisy_mag.unsqueeze(0).unsqueeze(0)  # (1, 1, F, T)
            predicted_mask = model(noisy_input)

            if predicted_mask.dim() == 4:
                predicted_mask = predicted_mask.squeeze(0).squeeze(0)
            elif predicted_mask.dim() == 3:
                predicted_mask = predicted_mask.squeeze(0)

            f_min = min(predicted_mask.shape[0], irm.shape[0])
            t_min = min(predicted_mask.shape[1], irm.shape[1])
            predicted_mask = predicted_mask[:f_min, :t_min]
            irm = irm[:f_min, :t_min]
            noisy_mag_t = noisy_mag[:f_min, :t_min]

            enhanced_mag = noisy_mag_t * predicted_mask
            clean_mag_t = clean_mag[:f_min, :t_min]
            l1_loss = nn.functional.l1_loss(enhanced_mag, clean_mag_t)

            noisy_phase_t = noisy_phase[:f_min, :t_min]
            enhanced_stft = enhanced_mag * torch.exp(1j * noisy_phase_t)
            full_enh_stft = torch.zeros_like(noisy_stft)
            full_enh_stft[:f_min, :t_min] = enhanced_stft
            enhanced_wav = torch.istft(full_enh_stft, frame_size, hop_size, window=window, length=len(clean), center=False)
            si_loss = si_sdr_loss(enhanced_wav, clean_t)

            loss = alpha * l1_loss + (1 - alpha) * si_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_train_loss = epoch_loss / max(len(train_loader), 1)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i in range(len(val_loader)):
                clean, primary, _, _ = val_loader[i]
                clean_t = torch.from_numpy(clean).float().to(device)
                primary_t = torch.from_numpy(primary).float().to(device)

                clean_stft = torch.stft(clean_t, frame_size, hop_size, window=window, return_complex=True, center=False)
                noisy_stft = torch.stft(primary_t, frame_size, hop_size, window=window, return_complex=True, center=False)

                noisy_mag = noisy_stft.abs()
                noisy_input = noisy_mag.unsqueeze(0).unsqueeze(0)
                pred_mask = model(noisy_input)

                if pred_mask.dim() == 4:
                    pred_mask = pred_mask.squeeze(0).squeeze(0)
                elif pred_mask.dim() == 3:
                    pred_mask = pred_mask.squeeze(0)

                f_min = min(pred_mask.shape[0], clean_stft.abs().shape[0])
                t_min = min(pred_mask.shape[1], clean_stft.abs().shape[1])

                enhanced_mag = noisy_mag[:f_min, :t_min] * pred_mask[:f_min, :t_min]
                clean_mag = clean_stft.abs()[:f_min, :t_min]
                v_loss = nn.functional.l1_loss(enhanced_mag, clean_mag)
                val_loss += v_loss.item()

        avg_val_loss = val_loss / max(len(val_loader), 1)
        epoch_time = time.perf_counter() - t0

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["epoch_time_s"].append(round(epoch_time, 2))

        print(f"  Epoch {epoch:03d}/{epochs:03d} | Train: {avg_train_loss:.4f} | "
              f"Val: {avg_val_loss:.4f} | Time: {epoch_time:.1f}s", flush=True)

    result = {
        "status": "COMPLETED",
        "model": model.__class__.__name__,
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "total_time_s": sum(history["epoch_time_s"]),
        "history": history,
    }

    # Save checkpoint + a run-metadata sidecar so this exact artifact is
    # traceable later (PH0.7 Gate 1 / Gate 4).
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
            "frame_size": frame_size,
            "hop_size": hop_size,
            "alpha": alpha,
            "device": device,
            "metadata_csv": str(meta_path.resolve()),
            "train_split": train_split,
            "train_samples": len(train_loader),
            "train_speakers": sorted(train_loader.speaker_ids()),
            "val_split": val_split,
            "val_samples": len(val_loader),
            "val_speakers": sorted(val_loader.speaker_ids()),
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss": history["val_loss"][-1],
            "torch_version": torch.__version__,
            "training_input": "RAW_PRIMARY",
            "deployment_input": "NLMS_RESIDUAL",
            "input_contract_status": "MISMATCH_PENDING_DESIGN",
        }
        sidecar_path = save_p.with_suffix(".json")
        with open(sidecar_path, "w", encoding="utf-8") as f:
            json.dump(run_metadata, f, indent=2)
        print(f"[+] Run metadata saved: {sidecar_path.resolve()}", flush=True)
        result["run_metadata_path"] = str(sidecar_path.resolve())

    return result


if __name__ == "__main__":
    if not TORCH_AVAILABLE:
        print("[!] PyTorch not available. Cannot train models.", flush=True)
        sys.exit(1)

    from src.ai.tiny_enhancer import TinyEnhancerNet

    print("=" * 60, flush=True)
    print("  TRAINING: TinyEnhancer V3", flush=True)
    print("=" * 60, flush=True)

    result = train_spectral_mask_model(
        model=TinyEnhancerNet(),
        epochs=30,
        lr=1e-3,
        save_path="checkpoints/tiny_enhancer_v3.pt",
    )
    print(f"\nResult: {result['status']}", flush=True)
