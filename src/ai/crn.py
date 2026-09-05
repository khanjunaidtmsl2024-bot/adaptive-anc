"""
CRN — Convolutional Recurrent Network for Real-Time Speech Enhancement.
PS 26052 — Adaptive Defence ANC.

Architecture:
  Encoder:  5× Conv2D(stride=2) blocks with BatchNorm + PReLU
  Bottleneck: 2-layer GRU on flattened frequency features
  Decoder:  5× ConvTranspose2D blocks with skip connections + BatchNorm + PReLU
  Output:   Conv2D → Sigmoid mask, applied in magnitude domain

Reference: Tan & Wang, "A Convolutional Recurrent Neural Network for
Real-Time Speech Enhancement", Interspeech 2018.

Parameter budget: ~550 K (Micro variant: ~55 K with reduced channels).
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

    class _EncoderBlock(nn.Module):
        def __init__(self, in_ch: int, out_ch: int, kernel: tuple = (3, 3), stride: tuple = (2, 1), pad: tuple = (1, 1)):
            super().__init__()
            self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, pad)
            self.bn = nn.BatchNorm2d(out_ch)
            self.act = nn.PReLU(out_ch)

        def forward(self, x):
            return self.act(self.bn(self.conv(x)))

    class _DecoderBlock(nn.Module):
        def __init__(self, in_ch: int, out_ch: int, kernel: tuple = (3, 3), stride: tuple = (2, 1),
                     pad: tuple = (1, 1), output_padding: tuple = (1, 0)):
            super().__init__()
            self.deconv = nn.ConvTranspose2d(in_ch, out_ch, kernel, stride, pad, output_padding=output_padding)
            self.bn = nn.BatchNorm2d(out_ch)
            self.act = nn.PReLU(out_ch)

        def forward(self, x):
            return self.act(self.bn(self.deconv(x)))


    class CRNNet(nn.Module):
        """
        Compact Convolutional Recurrent Network (Micro variant).

        Uses reduced channel widths [8, 16, 32, 64, 128] to keep params ~55K
        for real-time embedded deployment.

        Args:
            freq_bins:    Number of frequency bins (default 257 for frame_size=512).
            hidden_size:  GRU hidden units (default 128).
            channels:     Encoder/decoder channel progression.
        """

        def __init__(
            self,
            freq_bins: int = 257,
            hidden_size: int = 128,
            channels: Tuple[int, ...] = (8, 16, 32, 64, 128),
        ):
            super().__init__()
            self.freq_bins = freq_bins
            self.channels = channels

            # Encoder
            enc_layers = []
            in_ch = 1
            for out_ch in channels:
                enc_layers.append(_EncoderBlock(in_ch, out_ch))
                in_ch = out_ch
            self.encoder = nn.ModuleList(enc_layers)

            # Compute compressed freq dimension after encoder
            compressed_freq = freq_bins
            for _ in channels:
                compressed_freq = (compressed_freq + 1) // 2  # stride=2 in freq axis

            self.compressed_freq = compressed_freq
            gru_input_size = channels[-1] * compressed_freq

            # Bottleneck GRU
            self.gru = nn.GRU(
                input_size=gru_input_size,
                hidden_size=hidden_size,
                num_layers=2,
                batch_first=True,
                dropout=0.0,
            )
            self.gru_fc = nn.Linear(hidden_size, gru_input_size)

            # Decoder (with skip connections: concat doubles input channels)
            dec_layers = []
            for i in range(len(channels) - 1, 0, -1):
                # After concat: current channels + skip channels
                # Current channels: channels[i] for first block (from GRU), channels[i] for subsequent (from prev decoder)
                # Skip channels: channels[i-1] for first block, then channels[i-1] for each
                # Actually skip_idx = len(channels) - 2 - (iteration), so skip has channels[skip_idx]
                dec_in = channels[i] + channels[i - 1]  # concat(current, skip)
                dec_out = channels[i - 1]
                dec_layers.append(_DecoderBlock(dec_in, dec_out))
            self.decoder = nn.ModuleList(dec_layers)

            # Final projection to mask
            self.mask_conv = nn.Sequential(
                nn.ConvTranspose2d(channels[0] * 2, 1, kernel_size=(3, 3), stride=(2, 1),
                                   padding=(1, 1), output_padding=(1, 0)),
                nn.Sigmoid(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            x: (B, 1, F, T) magnitude spectrogram.
            Returns: (B, 1, F, T) estimated magnitude mask in [0, 1].
            """
            B, _, F, T = x.shape

            # Encode with skip connections
            skips = []
            h = x
            for enc in self.encoder:
                h = enc(h)
                skips.append(h)

            # GRU bottleneck: (B, C, F', T) → (B, T, C*F')
            _, C_last, F_comp, T_comp = h.shape
            h_flat = h.permute(0, 3, 1, 2).reshape(B, T_comp, C_last * F_comp)
            h_gru, _ = self.gru(h_flat)
            h_gru = self.gru_fc(h_gru)
            h = h_gru.reshape(B, T_comp, C_last, F_comp).permute(0, 2, 3, 1)  # (B, C, F', T)

            # Decode with skip connections
            for i, dec in enumerate(self.decoder):
                skip_idx = len(skips) - 2 - i  # skip from encoder
                skip = skips[skip_idx]

                # Align spatial dimensions
                if h.shape[2] != skip.shape[2]:
                    h = nn.functional.interpolate(h, size=(skip.shape[2], h.shape[3]))
                if h.shape[3] != skip.shape[3]:
                    h = nn.functional.interpolate(h, size=(h.shape[2], skip.shape[3]))

                h = torch.cat([h, skip], dim=1)
                h = dec(h)

            # Final mask projection
            skip0 = skips[0]
            if h.shape[2] != skip0.shape[2]:
                h = nn.functional.interpolate(h, size=(skip0.shape[2], h.shape[3]))
            if h.shape[3] != skip0.shape[3]:
                h = nn.functional.interpolate(h, size=(h.shape[2], skip0.shape[3]))

            h = torch.cat([h, skip0], dim=1)
            mask = self.mask_conv(h)

            # Ensure output matches input size
            if mask.shape[2] != F or mask.shape[3] != T:
                mask = nn.functional.interpolate(mask, size=(F, T), mode='bilinear', align_corners=False)

            return mask

        def count_parameters(self) -> int:
            return sum(p.numel() for p in self.parameters())

else:
    class CRNNet:
        def __init__(self, *a, **kw):
            raise ImportError("PyTorch required for CRNNet.")


class CRNWrapper:
    """Wraps CRNNet to conform to the spectral enhance_spectrogram interface."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "cpu",
        freq_bins: int = 257,
        hidden_size: int = 128,
        channels: Tuple[int, ...] = (8, 16, 32, 64, 128),
    ):
        self.device = device
        self.net = None

        if TORCH_AVAILABLE:
            self.net = CRNNet(
                freq_bins=freq_bins,
                hidden_size=hidden_size,
                channels=channels,
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
        mag_spec, phase: (freq_bins, time_frames).
        """
        if not TORCH_AVAILABLE or self.net is None:
            return mag_spec, phase

        with torch.no_grad():
            # (F, T) → (1, 1, F, T)
            tensor_in = torch.from_numpy(mag_spec).unsqueeze(0).unsqueeze(0).float().to(self.device)
            mask = self.net(tensor_in).squeeze().cpu().numpy()

            # Handle shape mismatch gracefully
            if mask.shape != mag_spec.shape:
                from scipy.ndimage import zoom
                zoom_factors = (mag_spec.shape[0] / mask.shape[0], mag_spec.shape[1] / mask.shape[1])
                mask = zoom(mask, zoom_factors, order=1)
                mask = np.clip(mask, 0, 1)

            enhanced_mag = mag_spec * mask

        return enhanced_mag.astype(np.float32), phase
