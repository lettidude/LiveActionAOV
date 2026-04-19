"""Pre-flight probe for DepthCrafter + NormalCrafter inference paths.

DepthCrafterPass / NormalCrafterPass were written against a *speculative*
diffusers API — both call `DiffusionPipeline.from_pretrained(model_id,
custom_pipeline=model_id, trust_remote_code=True)` and expect the pipeline
to accept `video=tensor, num_inference_steps=..., guidance_scale=...` and
return `.depth` / `.normals` / `.frames`. None of that was verified
against the real upstream repos. This probe cold-loads each pipeline,
inspects the signature, and runs one tiny window so we catch API drift
before a 100-frame job crashes mid-stream.

**Side effects on first run**:
- Downloads DepthCrafter (~10 GB, SVD backbone + adapter) into
  `~/.cache/huggingface/`.
- Downloads NormalCrafter (~10 GB, same structure) into
  `~/.cache/huggingface/`.

Each model is ~10 GB; budget disk + time accordingly.

Run::

    .venv/Scripts/python.exe scripts/probe_crafter_apis.py

Use `--skip-depth` / `--skip-normal` to probe one at a time. Non-zero exit
means one or both pipelines need code changes in the pass implementations.
"""

from __future__ import annotations

import argparse
import inspect
import sys
import traceback
from typing import Any

import numpy as np
import torch


def _rule(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _sig(obj: Any, name: str) -> None:
    try:
        s = inspect.signature(obj)
        print(f"  {name}{s}")
    except (TypeError, ValueError) as e:
        print(f"  {name}(?): signature unavailable — {e}")


def _describe_output(out: Any, prefix: str = "  ") -> None:
    """Walk whatever the pipeline returned and print shapes/dtypes of
    anything tensor-ish so we can wire postprocess() correctly."""
    seen_attrs = set()
    for name in ("depth", "normals", "frames", "images", "videos"):
        if hasattr(out, name):
            v = getattr(out, name)
            _describe_value(v, f"{prefix}out.{name}")
            seen_attrs.add(name)
    if not seen_attrs:
        # Maybe it's a plain tuple / list / tensor.
        _describe_value(out, f"{prefix}out")
        # Also dump public attrs in case the naming differs.
        for a in dir(out):
            if a.startswith("_") or a in seen_attrs:
                continue
            try:
                v = getattr(out, a)
            except Exception:
                continue
            if isinstance(v, (torch.Tensor, np.ndarray, list, tuple)):
                _describe_value(v, f"{prefix}out.{a}")


def _describe_value(v: Any, label: str) -> None:
    if isinstance(v, torch.Tensor):
        print(f"{label}: torch.Tensor shape={tuple(v.shape)} dtype={v.dtype}")
    elif isinstance(v, np.ndarray):
        print(f"{label}: ndarray shape={v.shape} dtype={v.dtype}")
    elif isinstance(v, (list, tuple)):
        print(f"{label}: {type(v).__name__} len={len(v)}")
        if v and isinstance(v[0], (torch.Tensor, np.ndarray)):
            _describe_value(v[0], f"{label}[0]")


def _load_pipeline(model_id: str) -> Any:
    """Try a few known paths to materialise the pipeline."""
    from diffusers import DiffusionPipeline

    last_err: Exception | None = None
    trials: list[tuple[str, dict[str, Any]]] = [
        (
            "fp16+use_safetensors, no custom_pipeline",
            dict(torch_dtype=torch.float16, use_safetensors=True),
        ),
        (
            "fp32+use_safetensors, no custom_pipeline",
            dict(torch_dtype=torch.float32, use_safetensors=True),
        ),
        (
            "fp16+use_safetensors+variant=fp16",
            dict(torch_dtype=torch.float16, use_safetensors=True, variant="fp16"),
        ),
    ]
    for i, (label, kwargs) in enumerate(trials, 1):
        print(f"  try {i}: {label}")
        try:
            pipe = DiffusionPipeline.from_pretrained(model_id, **kwargs)
            print(f"    OK — pipeline class: {type(pipe).__name__}")
            return pipe
        except Exception as e:
            last_err = e
            print(f"    FAILED: {type(e).__name__}: {str(e)[:200]}")

    raise RuntimeError(f"all load paths failed for {model_id}") from last_err


def probe_depthcrafter() -> Any:
    _rule("DepthCrafter — VIDEO_CLIP depth via SVD backbone")
    model_id = "tencent/DepthCrafter"
    print(f"Loading {model_id}...")
    pipe = _load_pipeline(model_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = pipe.to(device)
    print("\nPipeline attrs:")
    print(f"  type: {type(pipe).__name__}")
    _sig(pipe.__call__, "pipe.__call__")
    for attr in ("unet", "vae", "image_encoder", "scheduler"):
        if hasattr(pipe, attr):
            sub = getattr(pipe, attr)
            print(f"  pipe.{attr}: {type(sub).__name__}")

    # Feed a tiny 8-frame synthetic clip at low res to inspect the return.
    print("\nRunning tiny synthetic forward (8 frames × 256x384)...")
    n, h, w = 8, 256, 384
    # DepthCrafter wants float in [0, 1]; try ndarray first.
    video_np = np.random.rand(n, h, w, 3).astype(np.float32)
    for kw_name in ("video", "video_input", "image"):
        print(f"  trying kwarg {kw_name!r}...")
        try:
            with torch.no_grad():
                result = pipe(
                    **{kw_name: video_np},
                    num_inference_steps=1,
                    guidance_scale=1.0,
                    height=h,
                    width=w,
                )
            print(f"    OK. result type: {type(result).__name__}")
            _describe_output(result)
            return {"pipe": pipe, "device": device, "video_kwarg": kw_name}
        except Exception as e:
            print(f"    FAILED: {type(e).__name__}: {str(e)[:200]}")
            if (
                isinstance(e, (RuntimeError, torch.cuda.OutOfMemoryError))
                and "out of memory" in str(e).lower()
            ):
                raise
    print("  all known video kwargs failed — pipeline needs manual integration")
    return None


def probe_normalcrafter() -> Any:
    _rule("NormalCrafter — VIDEO_CLIP normals via SVD backbone")
    model_id = "Yanrui95/NormalCrafter"
    print(f"Loading {model_id}...")
    pipe = _load_pipeline(model_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = pipe.to(device)
    print("\nPipeline attrs:")
    print(f"  type: {type(pipe).__name__}")
    _sig(pipe.__call__, "pipe.__call__")
    for attr in ("unet", "vae", "image_encoder", "scheduler"):
        if hasattr(pipe, attr):
            sub = getattr(pipe, attr)
            print(f"  pipe.{attr}: {type(sub).__name__}")

    print("\nRunning tiny synthetic forward (8 frames × 256x384)...")
    n, h, w = 8, 256, 384
    video_np = np.random.rand(n, h, w, 3).astype(np.float32)
    for kw_name in ("video", "video_input", "image"):
        print(f"  trying kwarg {kw_name!r}...")
        try:
            with torch.no_grad():
                result = pipe(
                    **{kw_name: video_np},
                    num_inference_steps=1,
                    guidance_scale=1.0,
                    height=h,
                    width=w,
                )
            print(f"    OK. result type: {type(result).__name__}")
            _describe_output(result)
            return {"pipe": pipe, "device": device, "video_kwarg": kw_name}
        except Exception as e:
            print(f"    FAILED: {type(e).__name__}: {str(e)[:200]}")
            if (
                isinstance(e, (RuntimeError, torch.cuda.OutOfMemoryError))
                and "out of memory" in str(e).lower()
            ):
                raise
    print("  all known video kwargs failed — pipeline needs manual integration")
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-depth", action="store_true")
    ap.add_argument("--skip-normal", action="store_true")
    args = ap.parse_args()

    print("=" * 72)
    print("CRAFTER API PRE-FLIGHT PROBE")
    print("=" * 72)
    print(f"torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    try:
        import diffusers

        print(f"diffusers: {diffusers.__version__}")
    except ImportError:
        print("diffusers: NOT INSTALLED — run `pip install diffusers accelerate`")
        return 1

    dc = None
    nc = None
    if not args.skip_depth:
        try:
            dc = probe_depthcrafter()
        except Exception as e:
            print(f"\nDepthCrafter probe crashed: {type(e).__name__}: {e}")
            traceback.print_exc(limit=5)
    if not args.skip_normal:
        try:
            nc = probe_normalcrafter()
        except Exception as e:
            print(f"\nNormalCrafter probe crashed: {type(e).__name__}: {e}")
            traceback.print_exc(limit=5)

    _rule("SUMMARY")
    print(f"DepthCrafter  OK: {dc is not None}")
    print(f"NormalCrafter OK: {nc is not None}")
    if dc:
        print(f"  DepthCrafter video kwarg:  {dc['video_kwarg']!r}")
    if nc:
        print(f"  NormalCrafter video kwarg: {nc['video_kwarg']!r}")
    ok = (args.skip_depth or dc) and (args.skip_normal or nc)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
