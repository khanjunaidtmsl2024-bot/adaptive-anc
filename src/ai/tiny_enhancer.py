"""
TinyEnhancer PyTorch Architecture Adapter.
PS 26052 — Adaptive Defence ANC.

Implements the strictly causal 4-layer 2D ConvNet architecture (PH2A specification)
to enable local evaluation, streaming benchmarking, and supervisory auditing without
non-causal future temporal lookahead.
"""

from typing import Optional, Tuple, Dict, Any
import sys
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class CausalConv2d(nn.Module):
        """
        2D Convolution with strictly causal temporal padding and symmetric frequency padding.
        Input shape: (B, C, F, T)
        - Frequency (height): symmetric pad (kernel_size[0] - 1) // 2 on top and bottom.
        - Time (width): causal pad (kernel_size[1] - 1) on left (past), 0 on right (future).
        Supports stateful frame-by-frame streaming (T=1).
        """

        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Tuple[int, int] = (3, 3),
            bias: bool = True
        ):
            super().__init__()
            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size)
            self.k_f, self.k_t = kernel_size
            self.pad_f = (self.k_f - 1) // 2
            self.pad_t_past = self.k_t - 1
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0, bias=bias)
            self.register_buffer("state", None)

        def reset_state(self):
            self.state = None

        def forward(self, x: torch.Tensor, stateful: bool = False) -> torch.Tensor:
            B, C, F_dim, T_dim = x.shape
            if stateful:
                if self.state is None or self.state.shape[0] != B or self.state.shape[2] != F_dim:
                    self.state = torch.zeros(B, C, F_dim, self.pad_t_past, device=x.device, dtype=x.dtype)
                x_time = torch.cat([self.state, x], dim=-1)
                self.state = x_time[:, :, :, -self.pad_t_past:]
                x_pad = F.pad(x_time, (0, 0, self.pad_f, self.pad_f))
                return self.conv(x_pad)
            else:
                x_pad = F.pad(x, (self.pad_t_past, 0, self.pad_f, self.pad_f))
                return self.conv(x_pad)

    class TinyEnhancerNet(nn.Module):
        """4-layer strictly causal Conv2D mask estimator (PS 26052 PH2A contract)."""

        def __init__(self):
            super().__init__()
            self.conv1 = CausalConv2d(1, 16, kernel_size=(3, 3))
            self.relu1 = nn.ReLU()
            self.conv2 = CausalConv2d(16, 32, kernel_size=(3, 3))
            self.relu2 = nn.ReLU()
            self.conv3 = CausalConv2d(32, 16, kernel_size=(3, 3))
            self.relu3 = nn.ReLU()
            self.conv4 = CausalConv2d(16, 1, kernel_size=(3, 3))
            self.sigmoid = nn.Sigmoid()

        def reset_state(self):
            self.conv1.reset_state()
            self.conv2.reset_state()
            self.conv3.reset_state()
            self.conv4.reset_state()

        def forward(self, x: torch.Tensor, stateful: bool = False) -> torch.Tensor:
            """Estimate spectral suppression mask M(f, t) in [0, 1]."""
            x = self.relu1(self.conv1(x, stateful=stateful))
            x = self.relu2(self.conv2(x, stateful=stateful))
            x = self.relu3(self.conv3(x, stateful=stateful))
            x = self.sigmoid(self.conv4(x, stateful=stateful))
            return x

        def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
            """Loads state dict with transparent remapping of legacy Sequential keys."""
            new_state = {}
            for k, v in state_dict.items():
                k_new = k
                if k.startswith("network.0."):
                    k_new = k.replace("network.0.", "conv1.conv.")
                elif k.startswith("network.2."):
                    k_new = k.replace("network.2.", "conv2.conv.")
                elif k.startswith("network.4."):
                    k_new = k.replace("network.4.", "conv3.conv.")
                elif k.startswith("network.6."):
                    k_new = k.replace("network.6.", "conv4.conv.")
                new_state[k_new] = v
            return super().load_state_dict(new_state, strict=strict)

else:
    class TinyEnhancerNet:
        def __init__(self):
            raise ImportError("PyTorch is required to instantiate TinyEnhancerNet.")

TinyEnhancer = TinyEnhancerNet


class TinyEnhancerWrapper:
    """Wrapper that handles spectrogram conversion and inference."""

    def __init__(self, checkpoint_path: Optional[str] = None, device: str = "cpu"):
        self.device = device
        self.net = None
        self.checkpoint_loaded = False

        if TORCH_AVAILABLE:
            torch.manual_seed(42)
            self.net = TinyEnhancerNet().to(self.device)
            if checkpoint_path is None:
                from pathlib import Path
                default_ckpt = Path("checkpoints/tiny_enhancer_v3.pt")
                if default_ckpt.exists():
                    checkpoint_path = str(default_ckpt)

            if checkpoint_path is not None:
                try:
                    state_dict = torch.load(checkpoint_path, map_location=self.device)
                    self.net.load_state_dict(state_dict, strict=True)
                    self.checkpoint_loaded = True
                except Exception as e:
                    self.checkpoint_loaded = False
                    print(
                        f"[!] Warning: could not load checkpoint {checkpoint_path} ({e}); "
                        f"running with random weights. Check self.checkpoint_loaded before scoring.",
                        file=sys.stderr, flush=True,
                    )
            self.net.eval()

    def reset_state(self):
        """Reset temporal streaming state buffers."""
        if self.net is not None and hasattr(self.net, "reset_state"):
            self.net.reset_state()

    def enhance_spectrogram(
        self,
        mag_spec: np.ndarray,
        phase: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Takes 2D magnitude spectrogram, predicts mask, applies mask, and returns enhanced spec.
        Automatically activates stateful streaming when input is a single frame (T=1).
        """
        if not TORCH_AVAILABLE or self.net is None:
            return mag_spec, phase

        with torch.no_grad():
            orig_shape = mag_spec.shape
            is_single_frame = False
            if len(orig_shape) == 1:
                tensor_in = torch.from_numpy(mag_spec).unsqueeze(0).unsqueeze(0).unsqueeze(-1).float().to(self.device)
                is_single_frame = True
            elif len(orig_shape) == 2:
                tensor_in = torch.from_numpy(mag_spec).unsqueeze(0).unsqueeze(0).float().to(self.device)
                is_single_frame = (orig_shape[1] == 1)
            else:
                tensor_in = torch.from_numpy(mag_spec).float().to(self.device)
                is_single_frame = (tensor_in.shape[-1] == 1)

            mask_tensor = self.net(tensor_in, stateful=is_single_frame)
            mask = mask_tensor.squeeze(0).squeeze(0).cpu().numpy().reshape(orig_shape)
            enhanced_mag = mag_spec * mask

        return enhanced_mag.astype(np.float32), phase


class TinyEnhancerONNXWrapper:
    """
    Lightweight ONNX Runtime inference wrapper for E2_causal.onnx.
    Achieves sub-millisecond single-frame inference on CPU.
    """

    def __init__(self, onnx_path: Optional[str] = None, threads: int = 4):
        import onnxruntime as ort
        from pathlib import Path

        if onnx_path is None:
            for candidate in [
                Path("models/E2_causal.onnx"),
                Path(__file__).resolve().parent.parent.parent / "models" / "E2_causal.onnx",
                Path("deployment/rpi/models/E2_causal.onnx"),
            ]:
                if candidate.exists():
                    onnx_path = str(candidate)
                    break

        if onnx_path is None or not Path(onnx_path).exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.onnx_path = onnx_path

    def reset_state(self):
        """No-op for stateless sliding-window ONNX graph."""
        pass

    def enhance_spectrogram(
        self,
        mag_spec: np.ndarray,
        phase: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Runs single-frame or multi-frame inference using ONNX Runtime."""
        orig_shape = mag_spec.shape
        if len(orig_shape) == 1:
            inp = mag_spec.astype(np.float32)[np.newaxis, np.newaxis, :, np.newaxis]
        elif len(orig_shape) == 2:
            inp = mag_spec.astype(np.float32)[np.newaxis, np.newaxis, :, :]
        else:
            inp = mag_spec.astype(np.float32)

        mask = self.session.run(None, {self.input_name: inp})[0].reshape(orig_shape)
        enhanced_mag = mag_spec * mask
        return enhanced_mag.astype(np.float32), phase
