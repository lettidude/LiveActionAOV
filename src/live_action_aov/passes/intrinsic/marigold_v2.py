# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: MIT

"""Marigold V2 bridge passes — albedo and see-through depth via an EXTERNAL
environment (huawei-bayerlab/marigold-v2, Apache-2.0, ToG/SIGGRAPH Asia 2026).

Why a bridge and not an in-process model
----------------------------------------
Marigold V2 is a LoRA over a frozen 20B Qwen-Image-Edit-2509 DiT, run
through the authors' own framework with transformers 5.4 / diffusers 0.38
/ bitsandbytes pins that cannot coexist with this tool's venv. So the pass
shells out to the repo's `scripts/infer.py` in its own env:

    frames -> 8-bit sRGB PNGs (temp dir)            [this process]
           -> infer.py --modality ... --image_dir   [external env, ONE call
              per shot: the 4-bit quantisation of the DiT on load takes
              minutes, so never one process per frame]
           -> predictions_npy/*.npy                 [float32]
           -> channels                              [this process]

Setup (once, on disk A: per the house rule):
    git clone https://github.com/huawei-bayerlab/marigold-v2 A:/AI/marigold-v2
    bash A:/AI/marigold-v2/setup_win.sh     # uv venv + torch cu128 + assets

Point the pass at it with params `repo_dir` / `env_python` (defaults below).

Single-image model: NO temporal modelling. It is single-step (no stochastic
multi-step sampling, so the v1 noise-flicker mechanism is gone) but frame
coherence is whatever the per-image prediction gives — validated on real
plates before this pass ships as a default option. Heavy: ~17 GB VRAM at
1024², ~29 GB at 2048² — `inference_long_edge` caps the working size and
the prediction is resized back to the plate.

Two products:
- `marigold_v2_albedo`: linear-RGB albedo (lighting removed) -> albedo.rgb.
- `marigold_v2_depth_seethrough`: "layered" affine-invariant log depth that
  sees THROUGH glass/water (the checkpoint trained on LayeredDepth) ->
  Z / Z_raw, per-clip normalised like the other depth passes (1 = near).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from live_action_aov.core.pass_base import (
    ChannelSpec,
    License,
    PassType,
    TemporalMode,
    UtilityPass,
)
from live_action_aov.io.channels import (
    CH_ALBEDO_B,
    CH_ALBEDO_G,
    CH_ALBEDO_R,
    CH_Z,
    CH_Z_RAW,
)

_log = logging.getLogger(__name__)

_DEFAULT_REPO = Path(os.environ.get("LAAOV_MARIGOLD_V2_DIR", "A:/AI/marigold-v2"))


def _default_env_python(repo: Path) -> Path:
    sub = ("Scripts", "python.exe") if os.name == "nt" else ("bin", "python")
    return repo / ".venv" / sub[0] / sub[1]


def _resize(img: np.ndarray, w: int, h: int, *, area: bool) -> np.ndarray:
    import cv2

    interp = cv2.INTER_AREA if area else cv2.INTER_LINEAR
    return cv2.resize(img, (w, h), interpolation=interp)


def _fit_long_edge(w: int, h: int, long_edge: int) -> tuple[int, int]:
    """Working size: long edge capped, both dims multiples of 16 (the
    model's patch constraint — infer.py reshapes to /16 itself, but sizing
    here keeps the resize-back exact)."""
    s = min(1.0, long_edge / max(w, h))
    nw = max(16, int(round(w * s)) // 16 * 16)
    nh = max(16, int(round(h * s)) // 16 * 16)
    return nw, nh


class _MarigoldV2BridgeBase(UtilityPass):
    """Shared plumbing: PNG export, one subprocess per shot, npy import."""

    version = "0.1.0"
    license = License(
        spdx="Apache-2.0",
        commercial_use=True,
        commercial_tool_resale=True,
        notes=(
            "Marigold V2 code + weights are Apache-2.0 (Huawei Bayer Lab, 2026); "
            "the frozen Qwen-Image-Edit-2509 backbone is Apache-2.0 (Alibaba). "
            "Training datasets keep their own licences (weights only)."
        ),
    )
    temporal_mode = TemporalMode.VIDEO_CLIP  # run_shot batches the whole shot
    input_colorspace = "srgb_display"
    requires_artifacts: list[str] = []
    provides_artifacts: list[str] = []

    #: Subclass contract.
    modality: str = ""  # "albedo" | "depth" | "normals"
    checkpoint_subdir: str = ""  # "" = infer.py's default for the modality

    DEFAULT_PARAMS: dict[str, Any] = {
        "repo_dir": str(_DEFAULT_REPO),
        "env_python": "",  # "" = <repo_dir>/.venv/Scripts/python.exe
        "inference_long_edge": 1024,  # ~17 GB VRAM; 2048 needs ~29 GB
        "seed": 2025,
        "keep_temp": False,  # keep the PNG/npy exchange dir (debugging)
        "timeout_s": 4 * 3600,
    }

    # The model lives in another process — nothing to hold here, but the
    # executor's lifecycle expects a loader + the prefetch guard reads it.
    stub = False

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        for k, v in self.DEFAULT_PARAMS.items():
            self.params.setdefault(k, v)
        self._model: Any = None

    # ------------------------------------------------------------------
    # External env resolution
    # ------------------------------------------------------------------

    def _repo(self) -> Path:
        return Path(str(self.params["repo_dir"])).expanduser()

    def _python(self) -> Path:
        p = str(self.params.get("env_python") or "")
        return Path(p).expanduser() if p else _default_env_python(self._repo())

    def _check_env(self) -> None:
        repo, py = self._repo(), self._python()
        infer = repo / "scripts" / "infer.py"
        missing = [str(x) for x in (infer, py) if not x.is_file()]
        qwen = repo / "assets" / "checkpoints" / "Qwen-Image-Edit-2509"
        if not qwen.is_dir():
            missing.append(str(qwen))
        if missing:
            raise FileNotFoundError(
                "Marigold V2 external environment not found:\n  "
                + "\n  ".join(missing)
                + "\nSet it up once (git clone huawei-bayerlab/marigold-v2 + "
                "setup_win.sh + scripts/download_assets.py --skip-datasets) "
                "and point params.repo_dir / env_python at it."
            )

    def _load_model(self) -> None:
        self._check_env()
        self._model = object()

    # ------------------------------------------------------------------
    # Shot-level
    # ------------------------------------------------------------------

    def run_shot(
        self,
        reader: Any,
        frame_range: tuple[int, int],
    ) -> dict[int, dict[str, np.ndarray]]:
        import cv2

        self._load_model()
        first, last = frame_range
        long_edge = int(self.params["inference_long_edge"])
        work = Path(tempfile.mkdtemp(prefix=f"laaov_{self.name}_"))
        in_dir, out_dir = work / "in", work / "out"
        in_dir.mkdir()
        try:
            plate_shape: tuple[int, int] | None = None
            for f in range(first, last + 1):
                rgb = np.asarray(reader.read_frame(f)[0], dtype=np.float32)[..., :3]
                h, w = int(rgb.shape[0]), int(rgb.shape[1])
                plate_shape = (h, w)
                nw, nh = _fit_long_edge(w, h, long_edge)
                if (nw, nh) != (w, h):
                    rgb = _resize(rgb, nw, nh, area=True)
                u8 = (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
                cv2.imwrite(str(in_dir / f"f{f:08d}.png"), cv2.cvtColor(u8, cv2.COLOR_RGB2BGR))
            assert plate_shape is not None
            self._run_infer(in_dir, out_dir)
            return self._collect(out_dir, first, last, plate_shape)
        finally:
            if bool(self.params.get("keep_temp")):
                _log.info("%s: exchange dir kept at %s", self.name, work)
            else:
                shutil.rmtree(work, ignore_errors=True)

    def _infer_command(self, in_dir: Path, out_dir: Path) -> list[str]:
        repo = self._repo()
        cmd = [
            str(self._python()),
            str(repo / "scripts" / "infer.py"),
            "--modality",
            self.modality,
            "--image_dir",
            str(in_dir),
            "--output_dir",
            str(out_dir),
            "--seed",
            str(int(self.params["seed"])),
        ]
        if self.checkpoint_subdir:
            cmd += [
                "--checkpoint",
                str(repo / "assets" / "checkpoints" / "Marigold-V2" / self.checkpoint_subdir),
            ]
        return cmd

    def _run_infer(self, in_dir: Path, out_dir: Path) -> None:
        cmd = self._infer_command(in_dir, out_dir)
        _log.info("%s: %s", self.name, " ".join(cmd))
        env = dict(os.environ)
        env.setdefault("DEPTH_ASSETS_DIR", str(self._repo() / "assets"))
        proc = subprocess.run(
            cmd,
            cwd=str(self._repo()),
            env=env,
            capture_output=True,
            text=True,
            timeout=float(self.params["timeout_s"]),
            check=False,
        )
        if proc.returncode != 0:
            tail = "\n".join((proc.stderr or proc.stdout or "").splitlines()[-30:])
            raise RuntimeError(f"Marigold V2 inference failed (exit {proc.returncode}):\n{tail}")

    def _collect(
        self, out_dir: Path, first: int, last: int, plate_shape: tuple[int, int]
    ) -> dict[int, dict[str, np.ndarray]]:
        # infer.py nests the dataset name (out/<dataset>/predictions_npy/) —
        # search the whole output tree rather than pin that layout.
        preds = {p.stem: p for p in out_dir.rglob("*.npy") if p.parent.name == "predictions_npy"}
        h, w = plate_shape
        out: dict[int, dict[str, np.ndarray]] = {}
        for f in range(first, last + 1):
            key = f"f{f:08d}"
            p = preds.get(key)
            if p is None:
                raise RuntimeError(f"Marigold V2 produced no prediction for frame {f} ({key}.npy)")
            arr = np.load(p).astype(np.float32)
            out[f] = self._to_channels(arr, w, h)
        return self._finalize(out)

    # --- subclass hooks -----------------------------------------------

    def _to_channels(self, arr: np.ndarray, w: int, h: int) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def _finalize(self, out: dict[int, dict[str, np.ndarray]]) -> dict[int, dict[str, np.ndarray]]:
        return out

    # Per-frame lifecycle unused — run_shot is the entry.
    def preprocess(self, frames: np.ndarray) -> Any:
        return frames

    def infer(self, tensor: Any) -> Any:
        return tensor

    def postprocess(self, tensor: Any) -> dict[str, np.ndarray]:
        return {}


class MarigoldV2AlbedoPass(_MarigoldV2BridgeBase):
    name = "marigold_v2_albedo"
    pass_type = PassType.RADIOMETRIC
    modality = "albedo"

    produces_channels = [
        ChannelSpec(name=CH_ALBEDO_R, description="Albedo (base colour) R, lighting removed"),
        ChannelSpec(name=CH_ALBEDO_G, description="Albedo (base colour) G, lighting removed"),
        ChannelSpec(name=CH_ALBEDO_B, description="Albedo (base colour) B, lighting removed"),
    ]
    smoothable_channels: list[str] = []  # global drift handled here; see _finalize

    DEFAULT_PARAMS: dict[str, Any] = {
        **_MarigoldV2BridgeBase.DEFAULT_PARAMS,
        # Per-image model: measured on a real plate (PF0071, 16 frames) the
        # albedo of a static region drifts +-3-4.5% per channel frame to
        # frame, and 70% of the frame-to-frame change is a GLOBAL per-channel
        # gain. A per-frame gain match to the clip median removes it.
        "stabilize_gain": True,
        "stabilize_gain_clamp": 1.5,  # never scale a frame by more than this
    }

    def _finalize(self, out: dict[int, dict[str, np.ndarray]]) -> dict[int, dict[str, np.ndarray]]:
        if not bool(self.params.get("stabilize_gain", True)) or len(out) < 2:
            return out
        chans = (CH_ALBEDO_R, CH_ALBEDO_G, CH_ALBEDO_B)
        means = np.array([[float(c[ch].mean()) for ch in chans] for c in out.values()])
        ref = np.median(means, axis=0)
        clamp = float(self.params.get("stabilize_gain_clamp", 1.5))
        for c, m in zip(out.values(), means, strict=True):
            for k, ch in enumerate(chans):
                gain = float(np.clip(ref[k] / max(m[k], 1e-6), 1.0 / clamp, clamp))
                c[ch] = (c[ch] * gain).astype(np.float32)
        return out

    def _to_channels(self, arr: np.ndarray, w: int, h: int) -> dict[str, np.ndarray]:
        # infer.py saves albedo as [3, H, W] in sRGB (output_color_space=srgb).
        # Our albedo channels are LINEAR working space -> decode the EOTF.
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = np.transpose(arr, (1, 2, 0))
        srgb = np.clip(arr, 0.0, 1.0)
        lin = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4).astype(
            np.float32
        )
        if lin.shape[0] != h or lin.shape[1] != w:
            lin = _resize(lin, w, h, area=False)
        return {CH_ALBEDO_R: lin[..., 0], CH_ALBEDO_G: lin[..., 1], CH_ALBEDO_B: lin[..., 2]}


class MarigoldV2SeeThroughDepthPass(_MarigoldV2BridgeBase):
    name = "marigold_v2_depth_seethrough"
    pass_type = PassType.GEOMETRIC
    modality = "depth"
    checkpoint_subdir = "depth/Log-layered"  # see-through (LayeredDepth) log depth

    produces_channels = [
        ChannelSpec(
            name=CH_Z, description="See-through relative depth, per-clip normalized [0,1] (1=near)"
        ),
        ChannelSpec(name=CH_Z_RAW, description="Raw affine-invariant log depth (un-normalized)"),
    ]
    smoothable_channels = [CH_Z]

    DEFAULT_PARAMS: dict[str, Any] = {
        **_MarigoldV2BridgeBase.DEFAULT_PARAMS,
        "smooth": "auto",
        "depth_space": "relative",
    }

    def _to_channels(self, arr: np.ndarray, w: int, h: int) -> dict[str, np.ndarray]:
        if arr.ndim == 3:
            arr = arr[0]
        raw = arr.astype(np.float32)
        if raw.shape[0] != h or raw.shape[1] != w:
            raw = _resize(raw, w, h, area=False)
        return {CH_Z_RAW: raw}

    def _finalize(self, out: dict[int, dict[str, np.ndarray]]) -> dict[int, dict[str, np.ndarray]]:
        # Per-CLIP normalisation (not per frame — that would flicker), same
        # convention as the other depth passes: Z in [0,1], 1 = near. Log
        # depth grows with distance, so invert.
        stack = np.stack([c[CH_Z_RAW] for c in out.values()])
        lo, hi = float(np.percentile(stack, 0.5)), float(np.percentile(stack, 99.5))
        rng = max(hi - lo, 1e-6)
        for ch in out.values():
            ch[CH_Z] = (1.0 - np.clip((ch[CH_Z_RAW] - lo) / rng, 0.0, 1.0)).astype(np.float32)
        return out


__all__ = ["MarigoldV2AlbedoPass", "MarigoldV2SeeThroughDepthPass"]
