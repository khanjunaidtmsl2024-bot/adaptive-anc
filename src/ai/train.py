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
"""

import csv
import sys
import time
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
    """Loads DATASET_V004 samples for training."""

    def __init__(self, metadata_csv: str, data_root: str, split_filter: str = "TRAIN"):
        self.records = []
        with open(metadata_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
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


def train_spectral_mask_model(
    model: "nn.Module",
    metadata_csv: str = "data/v4/metadata/metadata_v4.csv",
    epochs: int = 30,
    lr: float = 1e-3,
    frame_size: int = 512,
    hop_size: int = 256,
    alpha: float = 0.7,
    device: str = "cpu",
    save_path: Optional[str] = None,
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
        save_path: Optional checkpoint save path.

    Returns:
        Training history dict.
    """
    if not TORCH_AVAILABLE:
        return {"status": "SKIPPED", "reason": "PyTorch not available"}

    meta_path = Path(metadata_csv)
    data_root = meta_path.parent.parent

    # Load training and validation data
    train_loader = DatasetLoader(str(meta_path), str(data_root), split_filter="train")
    val_loader = DatasetLoader(str(meta_path), str(data_root), split_filter="test_a")

    if len(train_loader) == 0:
        return {"status": "SKIPPED", "reason": "No training samples found"}

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    window = torch.hann_window(frame_size).to(device)

    history = {"train_loss": [], "val_loss": [], "epoch_time_s": []}

    print(f"\n[TRAIN] Model: {model.__class__.__name__}, Params: {sum(p.numel() for p in model.parameters()):,d}", flush=True)
    print(f"[TRAIN] Epochs: {epochs}, LR: {lr}, Device: {device}", flush=True)
    print(f"[TRAIN] Train samples: {len(train_loader)}, Val samples: {len(val_loader)}", flush=True)

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        epoch_loss = 0.0

        for i in range(len(train_loader)):
            clean, primary, reference, _ = train_loader[i]

            # Convert to tensors
            clean_t = torch.from_numpy(clean).float().to(device)
            primary_t = torch.from_numpy(primary).float().to(device)

            # STFT
            clean_stft = torch.stft(clean_t, frame_size, hop_size, window=window, return_complex=True)
            noisy_stft = torch.stft(primary_t, frame_size, hop_size, window=window, return_complex=True)

            clean_mag = clean_stft.abs()
            noisy_mag = noisy_stft.abs()
            noisy_phase = noisy_stft.angle()

            # Ideal ratio mask (IRM) as training target
            irm = clean_mag / (noisy_mag + 1e-8)
            irm = torch.clamp(irm, 0, 1)

            # Forward: model predicts mask from noisy magnitude
            # Input shape: (1, 1, F, T) for Conv2D models; handle different architectures
            noisy_input = noisy_mag.unsqueeze(0).unsqueeze(0)  # (1, 1, F, T)
            predicted_mask = model(noisy_input)

            if predicted_mask.dim() == 4:
                predicted_mask = predicted_mask.squeeze(0).squeeze(0)  # (F, T)
            elif predicted_mask.dim() == 3:
                predicted_mask = predicted_mask.squeeze(0)

            # Ensure shape match
            f_min = min(predicted_mask.shape[0], irm.shape[0])
            t_min = min(predicted_mask.shape[1], irm.shape[1])
            predicted_mask = predicted_mask[:f_min, :t_min]
            irm = irm[:f_min, :t_min]
            noisy_mag_t = noisy_mag[:f_min, :t_min]

            # L1 loss on masked magnitude
            enhanced_mag = noisy_mag_t * predicted_mask
            clean_mag_t = clean_mag[:f_min, :t_min]
            l1_loss = nn.functional.l1_loss(enhanced_mag, clean_mag_t)

            # Time-domain SI-SDR loss (reconstruct from masked STFT)
            noisy_phase_t = noisy_phase[:f_min, :t_min]
            enhanced_stft = enhanced_mag * torch.exp(1j * noisy_phase_t)
            # Pad back to full shape if needed
            full_enh_stft = torch.zeros_like(noisy_stft)
            full_enh_stft[:f_min, :t_min] = enhanced_stft
            enhanced_wav = torch.istft(full_enh_stft, frame_size, hop_size, window=window, length=len(clean))
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

                clean_stft = torch.stft(clean_t, frame_size, hop_size, window=window, return_complex=True)
                noisy_stft = torch.stft(primary_t, frame_size, hop_size, window=window, return_complex=True)

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

        print(f"  Epoch {epoch:03d}/{epochs:03d} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | Time: {epoch_time:.1f}s", flush=True)

    # Save checkpoint
    if save_path:
        save_p = Path(save_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), str(save_p))
        print(f"\n[+] Checkpoint saved: {save_p.resolve()}", flush=True)

    return {
        "status": "COMPLETED",
        "model": model.__class__.__name__,
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "total_time_s": sum(history["epoch_time_s"]),
        "history": history,
    }


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
