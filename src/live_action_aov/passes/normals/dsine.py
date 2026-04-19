"""DSINE normals pass (commercial-safe fallback, spec §13.1 Phase 2).

Backend: Baegwang-Bin DSINE (MIT weights). Loaded via `torch.hub.load`
(`baegwangbin/DSINE`, entry `DSINE`), which fetches + caches weights on first
use. `_load_model` is intentionally isolated so tests can monkeypatch a fake
model that returns known-shape tensors without downloading anything.

Outputs (spec §5.1, channels.py):
- `N.x`, `N.y`, `N.z` — camera-space, [-1, 1], **unit-length per pixel**
- `normals.confidence` — not emitted by DSINE; omitted for now

Spec §11.3 traps handled:
- **Trap 2 (renormalize after resize)**: bilinear upscale of unit-length
  normals breaks the unit constraint. We divide by `max(||N||, eps)` after
  the upscale, then clamp into [-1, 1].
- **Trap 3 (intrinsics scaling)**: DSINE consumes intrinsics to resolve
  the metric-to-pixel scale. When we run inference at a resized resolution,
  `fx, fy, cx, cy` must be scaled by `inf_size / plate_size` or the model
  sees a lens that doesn't match the image. Defaults fall back to a 50mm-
  equivalent `fx = fy = 0.8 * max(W, H)` when shot intrinsics are absent
  (design §8.4).

Axis convention:
- DSINE outputs OpenCV convention (+X right, +Y down, +Z forward into scene).
- Spec §10.3 locks +X right, +Y up, +Z toward camera (OpenGL/Maya convention).
- Conversion: flip Y and Z signs before writing channels.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from live_action_aov.core.pass_base import (
    ChannelSpec,
    License,
    PassType,
    TemporalMode,
    UtilityPass,
)
from live_action_aov.io.channels import CH_N_X, CH_N_Y, CH_N_Z


class DSINEPass(UtilityPass):
    name = "dsine"
    version = "0.1.0"
    license = License(
        spdx="MIT",
        commercial_use=True,
        commercial_tool_resale=True,
        notes="DSINE (Bae & Davison 2024). MIT-licensed weights.",
    )
    pass_type = PassType.GEOMETRIC
    temporal_mode = TemporalMode.PER_FRAME
    input_colorspace = "srgb_display"

    produces_channels = [
        ChannelSpec(name=CH_N_X, description="Camera-space normal x, [-1,1] unit-length"),
        ChannelSpec(name=CH_N_Y, description="Camera-space normal y"),
        ChannelSpec(name=CH_N_Z, description="Camera-space normal z"),
    ]
    # Smooth all three normal channels. The smoother renormalizes the triplet
    # back to unit length post-blend so Relight-node math stays correct.
    smoothable_channels = [CH_N_X, CH_N_Y, CH_N_Z]

    DEFAULT_PARAMS: dict[str, Any] = {
        "inference_short_edge": 480,   # DSINE default
        "precision": "fp32",           # "fp32" | "fp16" (fp16 only if cuda)
        "smooth": "auto",              # executor auto-wires TemporalSmoother
        # Intrinsics. `None` means "derive from shot/plate at inference time".
        "fx": None,
        "fy": None,
        "cx": None,
        "cy": None,
        # Axis convention. DSINE = "opencv", spec = "opengl". Postprocess flips
        # Y and Z when converting opencv → opengl.
        "input_axes": "opencv",
        "output_axes": "opengl",
    }

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        for k, v in self.DEFAULT_PARAMS.items():
            self.params.setdefault(k, v)
        self._model: Any = None
        self._device: Any = None
        self._dtype: Any = None

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        if self._model is not None:
            return
        import os
        import sys
        import types

        import torch

        # Upstream DSINE's `hubconf.py` does `from models import dsine;
        # dsine.DSINE()` — but the repo currently ships `models/dsine/` as a
        # package WITHOUT an `__init__.py` and the class is named
        # `DSINE_v02` in `models/dsine/v02.py`. `torch.hub.load(..., "DSINE")`
        # therefore raises `module 'models.dsine' has no attribute 'DSINE'`.
        # We sidestep the broken entry-point by (a) letting torch.hub fetch
        # and cache the repo, (b) adding its path to `sys.path`, and (c)
        # instantiating `DSINE_v02` directly with a minimal args namespace
        # (values mirror `projects/dsine/config.py` defaults).
        repo_dir = torch.hub._get_cache_or_reload(
            "baegwangbin/DSINE",
            force_reload=False,
            trust_repo=True,
            verbose=False,
            skip_validation=True,
        )
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)
        # The DSINE repo uses a bare `from models import dsine` import path,
        # which means it shadows anything else named `models` already in
        # `sys.modules` (e.g. from diffusers). Clear the stale entry first.
        for stale in [m for m in list(sys.modules) if m == "models" or m.startswith("models.")]:
            del sys.modules[stale]
        from models.dsine.v02 import DSINE_v02  # type: ignore

        args = types.SimpleNamespace(
            NNET_architecture="v02",
            NNET_output_dim=3,
            NNET_output_type="R",
            NNET_feature_dim=64,
            NNET_hidden_dim=64,
            NNET_encoder_B=5,
            NNET_decoder_NF=2048,
            NNET_decoder_BN=False,
            NNET_decoder_down=8,
            NNET_learned_upsampling=False,
            NRN_prop_ps=5,
            NRN_num_iter_train=5,
            NRN_num_iter_test=5,
            NRN_ray_relu=False,
        )
        model = DSINE_v02(args)

        # Load the pretrained checkpoint. torch.hub caches it under
        # `{hub_dir}/checkpoints/dsine.pt` — download if absent.
        ckpt_dir = os.path.join(torch.hub.get_dir(), "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, "dsine.pt")
        if not os.path.exists(ckpt_path):
            torch.hub.download_url_to_file(
                "https://huggingface.co/camenduru/DSINE/resolve/main/dsine.pt",
                ckpt_path,
                progress=True,
            )
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=True)

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = (
            torch.float16
            if self.params["precision"] == "fp16" and self._device.type == "cuda"
            else torch.float32
        )
        model.to(self._device).eval()
        # `pixel_coords` is a non-parameter buffer created on cuda:0 in __init__;
        # make sure it follows the model's actual device choice.
        model.pixel_coords = model.pixel_coords.to(self._device)
        if self._dtype == torch.float16:
            model.half()
        self._model = model

    # ------------------------------------------------------------------
    # Single-frame lifecycle (PER_FRAME)
    # ------------------------------------------------------------------

    def preprocess(self, frames: np.ndarray) -> Any:
        """Input: (1, H, W, 3) float32 sRGB-display in [0,1]. Returns the
        resized tensor + scaled intrinsics + plate shape."""
        import torch

        if frames.ndim != 4 or frames.shape[0] != 1 or frames.shape[-1] != 3:
            raise ValueError(
                f"DSINEPass preprocess expects (1, H, W, 3), got {frames.shape}"
            )
        self._load_model()
        plate_h, plate_w = int(frames.shape[1]), int(frames.shape[2])
        inf_h, inf_w = _inference_size(plate_h, plate_w, int(self.params["inference_short_edge"]))

        img = np.clip(frames[0], 0.0, 1.0).astype(np.float32, copy=False)
        t = torch.from_numpy(img).permute(2, 0, 1)[None]      # (1, 3, H, W)
        t = torch.nn.functional.interpolate(
            t, size=(inf_h, inf_w), mode="bilinear", align_corners=False
        )
        # ImageNet-like mean/std normalization — DSINE's forward expects this.
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        t = (t - mean) / std

        intrinsics = _scaled_intrinsics(
            plate_h, plate_w, inf_h, inf_w,
            fx=self.params["fx"], fy=self.params["fy"],
            cx=self.params["cx"], cy=self.params["cy"],
        )

        return {
            "image": t.to(self._device, dtype=self._dtype),
            "intrinsics": intrinsics.to(self._device, dtype=self._dtype),
            "plate_shape": (plate_h, plate_w),
            "inf_shape": (inf_h, inf_w),
        }

    def infer(self, tensor: Any) -> Any:
        import torch

        assert self._model is not None
        with torch.no_grad():
            pred = self._model(tensor["image"], intrins=tensor["intrinsics"])
        # DSINE's forward can return a list (one entry per scale) or a raw
        # tensor. Accept both — the last entry is always the finest-res output.
        if isinstance(pred, (list, tuple)):
            pred = pred[-1]
        return {
            "normals": pred.float(),
            "plate_shape": tensor["plate_shape"],
            "inf_shape": tensor["inf_shape"],
        }

    def postprocess(self, tensor: Any) -> dict[str, np.ndarray]:
        """Upscale to plate resolution, renormalize to unit length (trap 2),
        convert axis convention (opencv → opengl: flip Y and Z)."""
        import torch

        plate_h, plate_w = tensor["plate_shape"]
        n = tensor["normals"]
        if n.ndim != 4 or n.shape[1] != 3:
            raise ValueError(
                f"Expected normals (B, 3, h, w), got {tuple(n.shape)}"
            )
        n_up = torch.nn.functional.interpolate(
            n, size=(plate_h, plate_w), mode="bilinear", align_corners=False
        )
        # Trap 2: bilinear upscale breaks unit length — renormalize.
        mag = torch.sqrt((n_up ** 2).sum(dim=1, keepdim=True)).clamp_min(1e-6)
        n_unit = n_up / mag
        n_np = n_unit[0].cpu().numpy().astype(np.float32)   # (3, H, W)

        n_np = _convert_axes(
            n_np,
            src=str(self.params.get("input_axes", "opencv")),
            dst=str(self.params.get("output_axes", "opengl")),
        )
        # Safety clamp to [-1, 1] — guards against tiny FP overshoots.
        n_np = np.clip(n_np, -1.0, 1.0).astype(np.float32, copy=False)
        return {
            CH_N_X: n_np[0],
            CH_N_Y: n_np[1],
            CH_N_Z: n_np[2],
        }


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _inference_size(plate_h: int, plate_w: int, short_edge: int) -> tuple[int, int]:
    """Resize `plate_h x plate_w` so the shorter edge is `short_edge`, keeping
    aspect ratio. DSINE requires both dims to be multiples of 32 (see
    `utils.utils.get_pad` in the upstream repo — the encoder-decoder skip
    connections raise a `Sizes of tensors must match` error if either dim
    isn't a 32-multiple), so we round to 32 here rather than padding at
    forward time.
    """
    short = min(plate_h, plate_w)
    scale = short_edge / max(short, 1)
    inf_h = max(64, int(round(plate_h * scale / 32)) * 32)
    inf_w = max(64, int(round(plate_w * scale / 32)) * 32)
    return inf_h, inf_w


def _scaled_intrinsics(
    plate_h: int,
    plate_w: int,
    inf_h: int,
    inf_w: int,
    *,
    fx: float | None,
    fy: float | None,
    cx: float | None,
    cy: float | None,
) -> Any:
    """Build a (1, 3, 3) intrinsics matrix scaled for the inference resolution.

    Spec §11.3 trap 3. If any of fx/fy/cx/cy is None we fall back to a 50mm-
    equivalent approximation at plate resolution (`f = 0.8 * max(W, H)` per
    VFX convention), then scale to the inference resolution by the
    per-axis ratio.
    """
    import torch

    plate_fx = fx if fx is not None else 0.8 * max(plate_w, plate_h)
    plate_fy = fy if fy is not None else 0.8 * max(plate_w, plate_h)
    plate_cx = cx if cx is not None else plate_w / 2.0
    plate_cy = cy if cy is not None else plate_h / 2.0

    sx = inf_w / max(plate_w, 1)
    sy = inf_h / max(plate_h, 1)
    K = torch.tensor(
        [
            [plate_fx * sx, 0.0,           plate_cx * sx],
            [0.0,           plate_fy * sy, plate_cy * sy],
            [0.0,           0.0,           1.0],
        ],
        dtype=torch.float32,
    )[None]
    return K


def _convert_axes(n: np.ndarray, *, src: str, dst: str) -> np.ndarray:
    """Convert a (3, H, W) normals array between OpenCV and OpenGL axes.

    OpenCV: +X right, +Y down,  +Z forward (into scene, away from camera)
    OpenGL: +X right, +Y up,    +Z toward camera (out of screen)

    The conversion flips Y and Z — a double sign flip that leaves the
    handedness right-handed in both systems.
    """
    s = src.lower()
    d = dst.lower()
    if s == d:
        return n
    if {s, d} == {"opencv", "opengl"}:
        out = n.copy()
        out[1] *= -1.0
        out[2] *= -1.0
        return out
    raise ValueError(f"Unknown axis conversion {src!r} → {dst!r}")


__all__ = ["DSINEPass"]
