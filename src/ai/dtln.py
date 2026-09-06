"""
DTLN — Dual-signal Transformation LSTM Network.
PS 26052 — Adaptive Defence ANC.

Two-stage architecture:
  Stage 1:  STFT → |X| → LSTM₁ → σ(mask) → X⊙mask → iSTFT  (magnitude domain)
  Stage 2:  Conv1D encoder → LSTM₂ → Conv1D decoder              (learned features)

Reference: Westhausen & Meyer, "Dual-Signal Transformation LSTM Network
for Real-Time Noise Suppression", Interspeech 2020.

Parameter budget (measured, geometry-dependent):
  - frame 256 / hop 128 / enc 256 (PH1 contract): 775,939
  - frame 512 / hop 128 / enc 256 (legacy):       989,315
Defaults to the frozen PH1 contract geometry (frame 256, hop 128). Legacy
callers (offline phase4 benchmark) must pass frame_size=512 explicitly.
Internal torch.stft/istft use center=False (causal framing, no future
padding) matching the deployment STFT contract.
"""

from typing import Tuple, Optional
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    def _causal_stft(x: torch.Tensor, n_fft: int, hop: int, window: torch.Tensor) -> torch.Tensor:
        """Causal STFT (center=False semantics): frame t covers [t*hop, t*hop+n_fft).

        Returns complex spectrum of shape (B, n_fft//2+1, T_frames).
        """
        # unfold on (B, T) gives (B, T_f, n_fft); keep only complete frames
        n_frames = (x.shape[-1] - n_fft) // hop + 1
        frames = x.unfold(-1, n_fft, hop)[..., : n_frames, :]  # (B, T_f, n_fft)
        frames = frames * window          # (B, T_f, n_fft)
        return torch.fft.rfft(frames, dim=-1).transpose(1, 2)  # (B, F, T_f)

    def _causal_istft(spec: torch.Tensor, n_fft: int, hop: int, window: torch.Tensor,
                      length: int) -> torch.Tensor:
        """Causal WOLA iSTFT (center=False): window^2-normalized overlap-add.

        torch.istft(..., center=False) is broken on the pinned torch 2.14.0+cpu
        build (raises "window overlap add min"), so DTLN stage-1 synthesis uses
        this explicit overlap-add instead of silently keeping center=True
        (which would pad future samples and violate the causal contract).
        Interior reconstruction is exact to ~1e-6; the leading half-frame is
        under-determined by design (causal framing has no future context).
        """
        n_frames = spec.shape[-1]
        frames = torch.fft.irfft(spec, n_fft, dim=-2)          # (B, n_fft, T_f)
        frames = frames * window.view(1, -1, 1)                # synthesis window
        out_len = (n_frames - 1) * hop + n_fft
        y = torch.zeros(frames.shape[0], out_len, dtype=frames.dtype, device=frames.device)
        env = torch.zeros(out_len, dtype=frames.dtype, device=frames.device)
        for t in range(n_frames):
            s = t * hop
            y[:, s:s + n_fft] += frames[:, :, t]
            env[s:s + n_fft] += window ** 2
        y = y / (env + 1e-8)
        if y.shape[-1] >= length:
            return y[:, :length]
        return torch.nn.functional.pad(y, (0, length - y.shape[-1]))


    class _SeparationStage(nn.Module):
        """Single LSTM separation stage with layer-norm and FC projection."""

        def __init__(self, input_size: int, hidden_size: int, output_size: int):
            super().__init__()
            self.norm = nn.LayerNorm(input_size)
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=2,
                batch_first=True,
                dropout=0.0,
            )
            self.fc = nn.Linear(hidden_size, output_size)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """x: (B, T, F) → mask: (B, T, F) in [0, 1]."""
            x = self.norm(x)
            out, _ = self.lstm(x)
            return self.sigmoid(self.fc(out))


    class DTLNNet(nn.Module):
        """
        Dual-signal Transformation LSTM Network.

        Args:
            frame_size:  STFT window length. Default 256 = frozen PH1 contract
                         geometry; legacy frame-512 models pass 512.
            hop_size:    STFT hop length (default 128).
            hidden_size: LSTM hidden units per direction (default 128).
            encoder_size: Conv1D channels in Stage 2 (default 256).
        """

        def __init__(
            self,
            frame_size: int = 256,
            hop_size: int = 128,
            hidden_size: int = 128,
            encoder_size: int = 256,
        ):
            super().__init__()
            self.frame_size = frame_size
            self.hop_size = hop_size
            self.freq_bins = frame_size // 2 + 1  # 257 for 512

            # Stage 1: STFT magnitude mask estimation
            self.stage1 = _SeparationStage(
                input_size=self.freq_bins,
                hidden_size=hidden_size,
                output_size=self.freq_bins,
            )

            # Stage 2: Learned-feature domain refinement
            self.encoder = nn.Conv1d(1, encoder_size, kernel_size=frame_size, stride=hop_size, bias=False)
            self.stage2 = _SeparationStage(
                input_size=encoder_size,
                hidden_size=hidden_size,
                output_size=encoder_size,
            )
            self.decoder = nn.ConvTranspose1d(encoder_size, 1, kernel_size=frame_size, stride=hop_size, bias=False)

        def forward(self, noisy_wav: torch.Tensor) -> torch.Tensor:
            """
            End-to-end forward pass.

            Args:
                noisy_wav: (B, 1, T) waveform tensor.

            Returns:
                enhanced_wav: (B, 1, T) enhanced waveform tensor.
            """
            B, _, T = noisy_wav.shape

            # --- Stage 1: Magnitude-domain LSTM mask ---
            # Causal STFT (center=False semantics, matches deployment contract)
            wav_squeezed = noisy_wav.squeeze(1)  # (B, T)
            window = torch.hann_window(self.frame_size, device=noisy_wav.device)
            stft = _causal_stft(wav_squeezed, self.frame_size, self.hop_size, window)
            mag = stft.abs()         # (B, F, T_frames)
            phase = stft.angle()

            # LSTM mask
            mag_t = mag.permute(0, 2, 1)  # (B, T_frames, F)
            mask1 = self.stage1(mag_t)     # (B, T_frames, F)
            mask1 = mask1.permute(0, 2, 1) # (B, F, T_frames)

            # Apply mask and reconstruct time-domain (causal WOLA, center=False)
            stft_masked = (mag * mask1) * torch.exp(1j * phase)
            stage1_wav = _causal_istft(stft_masked, self.frame_size, self.hop_size,
                                       window, length=T).unsqueeze(1)  # (B, 1, T)

            # --- Stage 2: Learned-feature domain LSTM ---
            encoded = self.encoder(stage1_wav)     # (B, encoder_size, T_enc)
            encoded_t = encoded.permute(0, 2, 1)   # (B, T_enc, encoder_size)
            mask2 = self.stage2(encoded_t)          # (B, T_enc, encoder_size)
            refined = encoded_t * mask2
            refined = refined.permute(0, 2, 1)     # (B, encoder_size, T_enc)
            decoded = self.decoder(refined)         # (B, 1, T')

            # Trim/pad to original length
            if decoded.shape[-1] >= T:
                decoded = decoded[..., :T]
            else:
                decoded = nn.functional.pad(decoded, (0, T - decoded.shape[-1]))

            return decoded

        def count_parameters(self) -> int:
            return sum(p.numel() for p in self.parameters())

else:
    class DTLNNet:
        def __init__(self, *a, **kw):
            raise ImportError("PyTorch required for DTLNNet.")


class DTLNWrapper:
    """Wraps DTLNNet to conform to the spectral enhance_spectrogram interface."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "cpu",
        frame_size: int = 256,
        hop_size: int = 128,
        hidden_size: int = 128,
        encoder_size: int = 256,
    ):
        """
        frame_size defaults to 256 (frozen PH1 contract). Legacy offline models
        trained at frame_size=512 must pass 512 explicitly. The wrapper's own
        OLA iSTFT below uses the same frame/hop/window convention as the net.
        """
        self.device = device
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.net = None

        if TORCH_AVAILABLE:
            self.net = DTLNNet(
                frame_size=frame_size,
                hop_size=hop_size,
                hidden_size=hidden_size,
                encoder_size=encoder_size,
            ).to(device)
            if checkpoint_path is not None:
                import pathlib
                if pathlib.Path(checkpoint_path).exists():
                    state = torch.load(checkpoint_path, map_location=device)
                    self.net.load_state_dict(state)
            self.net.eval()

    def enhance_spectrogram(
        self,
        mag_spec: np.ndarray,
        phase: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interface-compatible spectral enhancement.
        Falls back to pass-through if torch unavailable.
        """
        if not TORCH_AVAILABLE or self.net is None:
            return mag_spec, phase

        # Reconstruct complex STFT → time-domain → run model → re-STFT
        complex_spec = mag_spec * np.exp(1j * phase)
        time_signal = np.fft.irfft(complex_spec, axis=0)

        # Flatten to 1D for the waveform model
        n_frames = time_signal.shape[1] if time_signal.ndim > 1 else 1
        if time_signal.ndim > 1:
            # Overlap-add reconstruction from spectrogram frames
            hop = self.hop_size
            total_len = self.frame_size + (n_frames - 1) * hop
            wav = np.zeros(total_len, dtype=np.float32)
            window = np.hanning(self.frame_size).astype(np.float32)
            norm = np.zeros(total_len, dtype=np.float32)
            for i in range(n_frames):
                s = i * hop
                frame = time_signal[:, i] * window
                wav[s:s + self.frame_size] += frame
                norm[s:s + self.frame_size] += window ** 2
            norm = np.maximum(norm, 1e-8)
            wav /= norm
        else:
            wav = time_signal.flatten()

        with torch.no_grad():
            tensor_in = torch.from_numpy(wav).unsqueeze(0).unsqueeze(0).float().to(self.device)
            enhanced = self.net(tensor_in).squeeze().cpu().numpy()

        # Convert back to spectrogram form
        window_np = np.hanning(self.frame_size).astype(np.float32)
        if len(enhanced) < self.frame_size:
            enhanced = np.pad(enhanced, (0, self.frame_size - len(enhanced)))

        n_out_frames = max(1, (len(enhanced) - self.frame_size) // self.hop_size + 1)
        out_frames = np.zeros((self.frame_size, n_out_frames), dtype=np.float32)
        for i in range(n_out_frames):
            s = i * self.hop_size
            chunk = enhanced[s:s + self.frame_size]
            if len(chunk) < self.frame_size:
                chunk = np.pad(chunk, (0, self.frame_size - len(chunk)))
            out_frames[:, i] = chunk * window_np

        out_stft = np.fft.rfft(out_frames, axis=0)
        out_mag = np.abs(out_stft).astype(np.float32)
        out_phase = np.angle(out_stft).astype(np.float32)

        # Match original shape
        target_f, target_t = mag_spec.shape
        if out_mag.shape[0] > target_f:
            out_mag = out_mag[:target_f, :]
            out_phase = out_phase[:target_f, :]
        if out_mag.shape[1] > target_t:
            out_mag = out_mag[:, :target_t]
            out_phase = out_phase[:, :target_t]
        if out_mag.shape[1] < target_t:
            pad_t = target_t - out_mag.shape[1]
            out_mag = np.pad(out_mag, ((0, 0), (0, pad_t)))
            out_phase = np.pad(out_phase, ((0, 0), (0, pad_t)))

        return out_mag, out_phase

    def enhance_waveform(self, wav: np.ndarray) -> np.ndarray:
        """Direct waveform-in → waveform-out enhancement."""
        if not TORCH_AVAILABLE or self.net is None:
            return wav

        with torch.no_grad():
            tensor_in = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)
            enhanced = self.net(tensor_in).squeeze().cpu().numpy()

        if len(enhanced) > len(wav):
            enhanced = enhanced[:len(wav)]
        elif len(enhanced) < len(wav):
            enhanced = np.pad(enhanced, (0, len(wav) - len(enhanced)))

        return enhanced.astype(np.float32)
