"""NormalCrafter pass (VIDEO_CLIP, temporal-native normals).

Backend: the vendored NormalCrafter pipeline from
github.com/Binyr/NormalCrafter plus the `Yanrui95/NormalCrafter` weights
on HuggingFace. Built on Stable Video Diffusion's UNet backbone, retrained
by Binyr et al. to output per-frame camera-space normals instead of
colour frames. Runtime: a handful of minutes for ~100 frames on a 5090
(one denoising step per window by default).

Why we vendor the upstream pipeline instead of loading via
`DiffusionPipeline.from_pretrained`
---------------------------------------------------------------------

The probe (`scripts/probe_crafter_apis.py`) found that HF's auto-pipeline
resolves `Yanrui95/NormalCrafter` to a vanilla `StableVideoDiffusionPipeline`
— the custom `NormalCrafterPipeline` class (with temporal windowing and a
normal-specific `__call__`) lives only in the Binyr GitHub repo. We fetch
that source into `src/live_action_aov/vendored/normalcrafter/` (MIT-
licensed, kept verbatim with its LICENSE alongside) and load through it.

Licence
-------

NormalCrafter's own code is MIT and the retrained weights are Apache-2.0,
but the UNet inherits Stable Video Diffusion's architecture + initial
weights — so the combined package is effectively SVD-NC (non-commercial).
For internal pipelines that's fine. The pass declares
`commercial_use=False` and the CLI gates it behind
`--allow-noncommercial`.

Outputs (spec §5.1, channels.py)
--------------------------------

Camera-space unit normals packed as three scalar channels:
- `normal.x`, `normal.y`, `normal.z`  (each ∈ [-1, 1])

Values are unit-length per pixel (spec trap 2). Axis convention:
NormalCrafter emits in OpenCV camera space (+X right, +Y **down**,
+Z forward-into-scene). Spec §10.3 requires OpenGL/Maya (+X right,
+Y **up**, +Z toward camera). We flip Y and Z after inference — same
helper DSINE uses, so downstream tooling behaviour is consistent
whether a shot was run through DSINE or NormalCrafter.

Temporal: VIDEO_CLIP. The pipeline does its own sliding-window inference
(default `window_size=14`, `time_step_size=10`, i.e. 4-frame overlap per
14-frame window) with linear-merge blending across overlaps, so there's
nothing for an external smoother to do. `smoothable_channels = []`.

Implementation notes
--------------------

- The pipeline wants a list of `PIL.Image` (H, W, 3) in [0, 255], not a
  numpy array. `utils.read_video_frames` in the upstream repo converts
  from decord frames to PIL before the call — we mirror that: convert
  display-transformed float32 [0, 1] plate → uint8 → list of PIL.
- `max_res` caps the longer edge at inference time (default 1024) to
  keep VRAM bounded. The pipeline itself pads to multiples of 64
  internally and returns unpadded frames, so we only need to handle
  the downscale-back-to-plate step here.
- Tests bypass the real diffusers pipeline by subclassing and
  overriding `_infer_clip`, which is the single hook between the
  I/O boilerplate in `run_shot` and the real SVD denoising loop.
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
from live_action_aov.passes.normals.dsine import _convert_axes


class NormalCrafterPass(UtilityPass):
    name = "normalcrafter"
    version = "0.2.0"
    license = License(
        spdx="MIT+Apache-2.0+SVD-NC",
        commercial_use=False,
        commercial_tool_resale=False,
        notes=(
            "NormalCrafter is built on Stable Video Diffusion, whose "
            "commercial terms are non-permissive. NormalCrafter's own "
            "code is MIT and the retrained weights are Apache-2.0, but "
            "the package inherits SVD-NC restrictions. Gated behind "
            "--allow-noncommercial."
        ),
    )
    pass_type = PassType.GEOMETRIC
    temporal_mode = TemporalMode.VIDEO_CLIP
    temporal_window = 14
    input_colorspace = "srgb_display"

    produces_channels = [
        ChannelSpec(name=CH_N_X, description="Camera-space normal x (OpenGL), [-1, 1] unit-length"),
        ChannelSpec(name=CH_N_Y, description="Camera-space normal y (OpenGL)"),
        ChannelSpec(name=CH_N_Z, description="Camera-space normal z (OpenGL)"),
    ]
    smoothable_channels: list[str] = []  # VIDEO_CLIP: internally temporally coherent

    DEFAULT_PARAMS: dict[str, Any] = {
        "model_id": "Yanrui95/NormalCrafter",
        "precision": "fp16",
        "window_size": 14,  # pipeline's internal sliding window
        "time_step_size": 10,  # stride between windows (overlap = window - stride)
        "decode_chunk_size": 7,  # VAE decode chunk size (memory vs speed)
        "max_res": 1024,  # cap longer edge for VRAM
        "seed": 42,
        "cpu_offload": None,  # None | "model" | "sequential"
        # Axis convention. First run on CAT_070_0030 showed NormalCrafter
        # emits normals where visible-surface Z is ALREADY positive (i.e.
        # +Z toward camera — OpenGL convention). Applying an OpenCV→OpenGL
        # flip on top turned every face yellow-green (nz ∈ [-1, -0.1])
        # instead of the expected cyan (nz ∈ [+0.1, +1]). So the input is
        # OpenGL-native and we skip conversion by default.
        "input_axes": "opengl",
        "output_axes": "opengl",
        # Legacy knobs kept for parameter-compatibility with old YAMLs and
        # tests; validated here but not otherwise used (pipeline handles
        # internal windowing through window_size / time_step_size above).
        "window": 14,
        "overlap": 4,
    }

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        for k, v in self.DEFAULT_PARAMS.items():
            self.params.setdefault(k, v)
        if int(self.params["overlap"]) >= int(self.params["window"]):
            raise ValueError(
                f"overlap ({self.params['overlap']}) must be < window ({self.params['window']})"
            )
        self._pipeline: Any = None
        self._device: Any = None
        self._dtype: Any = None
        self._frame_keys: list[int] = []
        self._plate_shape: tuple[int, int] = (0, 0)

    @classmethod
    def declared_license(cls) -> License:
        """Shortcut used by tests / audits. Mirrors `cls.license`."""
        return cls.license

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        if self._pipeline is not None:
            return
        try:
            import diffusers  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "NormalCrafter requires the `diffusers` package. "
                "Install via: pip install live-action-aov[normalcrafter]"
            ) from e
        import torch
        from diffusers import AutoencoderKLTemporalDecoder

        from live_action_aov.vendored.normalcrafter.normal_crafter_ppl import (
            NormalCrafterPipeline,
        )
        from live_action_aov.vendored.normalcrafter.unet import (
            DiffusersUNetSpatioTemporalConditionModelNormalCrafter,
        )

        model_id = str(self.params["model_id"])
        use_fp16 = str(self.params["precision"]).lower() == "fp16"
        weight_dtype = torch.float16 if use_fp16 else torch.float32

        # The custom UNet subclass adds NormalCrafter-specific overrides;
        # HF's auto-loader would otherwise instantiate the vanilla
        # `UNetSpatioTemporalConditionModel` and miss them.
        unet = DiffusersUNetSpatioTemporalConditionModelNormalCrafter.from_pretrained(
            model_id,
            subfolder="unet",
            low_cpu_mem_usage=True,
        )
        vae = AutoencoderKLTemporalDecoder.from_pretrained(
            model_id,
            subfolder="vae",
        )
        unet.to(dtype=weight_dtype)
        vae.to(dtype=weight_dtype)

        pipe = NormalCrafterPipeline.from_pretrained(
            model_id,
            unet=unet,
            vae=vae,
            torch_dtype=weight_dtype,
            variant="fp16" if use_fp16 else None,
        )

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = weight_dtype

        offload = self.params.get("cpu_offload")
        if offload == "sequential":
            pipe.enable_sequential_cpu_offload()
        elif offload == "model":
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(self._device)
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            # xFormers is optional — slower without it, otherwise fine.
            pass

        self._pipeline = pipe

    # ------------------------------------------------------------------
    # Stubs — the UtilityPass base still expects these; VIDEO_CLIP drives
    # everything through run_shot.
    # ------------------------------------------------------------------

    def preprocess(self, frames: np.ndarray) -> Any:
        return frames

    def infer(self, tensor: Any) -> Any:
        raise NotImplementedError("NormalCrafterPass is VIDEO_CLIP; drive it via run_shot.")

    def postprocess(self, tensor: Any) -> dict[str, np.ndarray]:
        raise NotImplementedError("NormalCrafterPass is VIDEO_CLIP; drive it via run_shot.")

    # ------------------------------------------------------------------
    # Shot-level: one pipeline call across the whole clip.
    # ------------------------------------------------------------------

    def run_shot(
        self,
        reader: Any,
        frame_range: tuple[int, int],
    ) -> dict[int, dict[str, np.ndarray]]:
        first, last = frame_range

        # 1. Read plate frames into a (N, H, W, 3) float32 [0, 1] stack.
        frames_float: list[np.ndarray] = []
        for f in range(first, last + 1):
            arr, _ = reader.read_frame(f)
            frames_float.append(np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False))
        stack_f = np.stack(frames_float, axis=0)
        plate_h, plate_w = int(stack_f.shape[1]), int(stack_f.shape[2])

        # 2. Resize to exactly 576 × 1024 — the only resolution SVD-xt's
        #    UNet accepts. Earlier attempts preserved plate aspect via
        #    long-edge scaling + /64 rounding, but that crashed on any
        #    plate TALLER than 16:9 (e.g. 2048 × 1408 open-gate produces
        #    704 × 1024 which the UNet rejects with
        #        "Expected size 72 but got size 88 for tensor number 1".
        #    The UNet has zero tolerance for anything off-spec).
        #
        #    Normals are unit vectors so non-uniform stretch at the
        #    model input distorts the per-pixel direction; we reverse
        #    the stretch when we bilinear-upscale back to plate res
        #    and renormalise to unit length afterwards (step 5 below),
        #    which takes care of the per-pixel magnitude regardless of
        #    how lossy the resize chain was. Visual quality is
        #    indistinguishable from aspect-preserving letterbox in
        #    practice; a proper letterbox path is a TODO.
        inf_h = 576
        inf_w = 1024
        if (inf_h, inf_w) != (plate_h, plate_w):
            import cv2

            rs = np.empty((stack_f.shape[0], inf_h, inf_w, 3), dtype=np.float32)
            for i in range(stack_f.shape[0]):
                rs[i] = cv2.resize(stack_f[i], (inf_w, inf_h), interpolation=cv2.INTER_AREA)
            stack_f = rs

        # 3. Convert to uint8 + PIL (what the pipeline expects). See
        #    `read_video_frames` upstream — the pipeline's pad + _encode_image
        #    paths branch on PIL vs ndarray and the PIL branch is the
        #    battle-tested one.
        stack_u8 = (np.clip(stack_f, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        frames_pil = self._to_pil(stack_u8)

        # 4. Inference (single hook so tests can bypass).
        normals = self._infer_clip(frames_pil)  # (N, H, W, 3) in [-1, 1]

        # 5. Upscale back to plate res if we downscaled earlier, then
        #    renormalise unit length (spec trap 2 — bilinear blend breaks
        #    it even when each pixel was unit before).
        if normals.shape[1] != plate_h or normals.shape[2] != plate_w:
            import cv2

            up = np.empty((normals.shape[0], plate_h, plate_w, 3), dtype=np.float32)
            for i in range(normals.shape[0]):
                up[i] = cv2.resize(
                    normals[i].astype(np.float32, copy=False),
                    (plate_w, plate_h),
                    interpolation=cv2.INTER_LINEAR,
                )
            normals = up
        mag = np.sqrt((normals**2).sum(axis=-1, keepdims=True))
        mag = np.maximum(mag, 1e-6)
        normals = normals / mag

        # 6. Axis convention opencv → opengl (default).
        src = str(self.params.get("input_axes", "opencv"))
        dst = str(self.params.get("output_axes", "opengl"))
        if src.lower() != dst.lower():
            converted = np.empty_like(normals)
            for i in range(normals.shape[0]):
                n = normals[i].transpose(2, 0, 1)  # (3, H, W)
                n = _convert_axes(n, src=src, dst=dst)
                converted[i] = n.transpose(1, 2, 0)
            normals = converted

        normals = np.clip(normals, -1.0, 1.0).astype(np.float32, copy=False)

        # 7. Per-frame dict keyed by plate frame number.
        self._frame_keys = list(range(first, last + 1))
        self._plate_shape = (plate_h, plate_w)
        out: dict[int, dict[str, np.ndarray]] = {}
        for i, f in enumerate(self._frame_keys):
            out[f] = {
                CH_N_X: normals[i, :, :, 0],
                CH_N_Y: normals[i, :, :, 1],
                CH_N_Z: normals[i, :, :, 2],
            }
        return out

    # ------------------------------------------------------------------
    # Inference hook — overridden in tests.
    # ------------------------------------------------------------------

    def _to_pil(self, stack_u8: np.ndarray) -> Any:
        """(N, H, W, 3) uint8 → list of PIL.Image."""
        from PIL import Image

        return [Image.fromarray(stack_u8[i]) for i in range(stack_u8.shape[0])]

    def _infer_clip(self, frames_pil: Any) -> np.ndarray:
        """Drive the vendored pipeline on the full clip.

        Returns (N, H, W, 3) float32 normals in [-1, 1], in the pipeline's
        native OpenCV camera space. Tests override this to skip diffusers.
        """
        import torch

        self._load_model()
        assert self._pipeline is not None

        generator = torch.Generator(device=self._device).manual_seed(int(self.params["seed"]))
        with torch.inference_mode():
            result = self._pipeline(
                frames_pil,
                decode_chunk_size=int(self.params["decode_chunk_size"]),
                time_step_size=int(self.params["time_step_size"]),
                window_size=int(self.params["window_size"]),
                generator=generator,
            )
        frames = result.frames[0]  # (N, H, W, 3), range (-1, 1)
        return np.asarray(frames, dtype=np.float32)


__all__ = ["NormalCrafterPass"]
