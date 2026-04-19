"""Pre-flight probe for SAM 3 + RVM real-weights inference paths.

Run this once before implementing the three NotImplementedError stubs in
`passes/matte/sam3.py` and `passes/matte/rvm.py`. It loads each real model
and prints the API surface we plan to call — class names, method
signatures, return shapes/dtypes — so we catch upstream API drift before
spending an hour debugging wrong assumptions.

**Side effects on first run**:
- Downloads SAM 3 (~3.44 GB) into `~/.cache/huggingface/`.
- Downloads RVM MobileNetV3 (~14 MB) into `~/.cache/torch/hub/`.

Run::

    .venv/Scripts/python.exe scripts/probe_matte_apis.py

This file is intentionally NOT in the test suite — it's a throwaway probe
to re-run whenever `transformers` or `torch.hub` update and we want to
confirm our assumptions still hold. Keep it under version control so the
next person (or next Claude) doesn't have to re-derive it.
"""

from __future__ import annotations

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
    """Print the callable's signature, swallowing errors so one missing
    method doesn't abort the probe."""
    try:
        s = inspect.signature(obj)
        print(f"  {name}{s}")
    except (TypeError, ValueError) as e:
        print(f"  {name}(?): signature unavailable — {e}")


def probe_sam3_detector() -> Any:
    """Load SAM 3, run one forward on a synthetic image, report shapes."""
    _rule("SAM 3 — single-frame detector")
    try:
        from transformers import AutoModel, AutoProcessor
    except ImportError as e:
        print(f"FAIL: transformers import failed — {e}")
        return None

    repo = "facebook/sam3"
    print(f"Loading processor from {repo}...")
    processor = AutoProcessor.from_pretrained(repo)
    print(f"  processor class: {type(processor).__name__}")

    print(f"Loading model from {repo}...")
    model = AutoModel.from_pretrained(repo, trust_remote_code=True)
    print(f"  model class: {type(model).__name__}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")
    model = model.to(device).eval()

    print("\nProcessor inspection:")
    # Is post_process_instance_segmentation the right name?
    pp_candidates = [
        "post_process_instance_segmentation",
        "post_process_masks",
        "post_process_for_mask_generation",
    ]
    for name in pp_candidates:
        if hasattr(processor, name):
            _sig(getattr(processor, name), f"processor.{name}")

    # __call__ signature — shows the text= / images= kwargs we rely on.
    _sig(processor.__call__, "processor.__call__")

    print("\nRunning one forward pass on a 256x384 synthetic image with text='person'...")
    # Grey image so the detector almost certainly finds nothing — we only care
    # about the output *shapes*, not content.
    img = np.full((256, 384, 3), 128, dtype=np.uint8)
    inputs = processor(images=img, text="person", return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    print(f"  output type: {type(outputs).__name__}")
    print(
        f"  output keys/attrs: {list(outputs.keys()) if hasattr(outputs, 'keys') else dir(outputs)[:15]}"
    )

    # Post-process — find the right method name dynamically.
    target_sizes_kw = None
    for candidate in ("original_sizes", "target_sizes"):
        if candidate in inputs:
            target_sizes_kw = candidate
            break
    if target_sizes_kw:
        print(f"  inputs has '{target_sizes_kw}': {inputs[target_sizes_kw]}")

    pp_method = None
    for name in pp_candidates:
        if hasattr(processor, name):
            pp_method = getattr(processor, name)
            print(f"\nCalling processor.{name}(...)...")
            try:
                if name == "post_process_instance_segmentation":
                    results = pp_method(
                        outputs,
                        threshold=0.5,
                        mask_threshold=0.5,
                        target_sizes=[(256, 384)],
                    )
                elif name == "post_process_masks":
                    results = pp_method(
                        outputs.pred_masks
                        if hasattr(outputs, "pred_masks")
                        else outputs["pred_masks"],
                        original_sizes=[(256, 384)],
                    )
                else:
                    results = pp_method(outputs)
                print(
                    f"  results type: {type(results).__name__}; len: {len(results) if hasattr(results, '__len__') else '?'}"
                )
                if isinstance(results, list) and results:
                    first = results[0]
                    print(f"  results[0] type: {type(first).__name__}")
                    if isinstance(first, dict):
                        for k, v in first.items():
                            shape = getattr(v, "shape", None)
                            dtype = getattr(v, "dtype", None)
                            print(f"    results[0][{k!r}]: shape={shape}, dtype={dtype}")
            except Exception as e:
                print(f"  {name} RAISED: {type(e).__name__}: {e}")
                traceback.print_exc(limit=3)
            break
    if pp_method is None:
        print("  WARN: no known post-process method found on processor.")

    return {"model": model, "processor": processor, "device": device}


def probe_sam3_tracker() -> Any:
    """Load the SAM 3 video-tracker classes and inspect the session API."""
    _rule("SAM 3 — video tracker (PVS head)")
    try:
        from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor

        print("Importing Sam3TrackerVideoModel / Sam3TrackerVideoProcessor: OK")
    except ImportError as e:
        print(f"FAIL: tracker classes not importable — {e}")
        print("  Falling back: check if they're under trust_remote_code classes on the model card.")
        return None

    repo = "facebook/sam3"
    print(f"Loading tracker processor from {repo}...")
    try:
        proc = Sam3TrackerVideoProcessor.from_pretrained(repo)
        print(f"  tracker processor class: {type(proc).__name__}")
    except Exception as e:
        print(f"FAIL loading tracker processor: {type(e).__name__}: {e}")
        return None

    print("\nTracker processor method signatures:")
    for name in (
        "init_video_session",
        "add_inputs_to_inference_session",
        "post_process_masks",
    ):
        if hasattr(proc, name):
            _sig(getattr(proc, name), f"proc.{name}")
        else:
            print(f"  proc.{name}: MISSING")

    print("\nKey question: does add_inputs_to_inference_session accept input_masks=?")
    if hasattr(proc, "add_inputs_to_inference_session"):
        sig = inspect.signature(proc.add_inputs_to_inference_session)
        has_masks = "input_masks" in sig.parameters
        has_points = "input_points" in sig.parameters
        print(f"  input_masks kwarg:  {'YES' if has_masks else 'NO (use centroid point fallback)'}")
        print(f"  input_points kwarg: {'YES' if has_points else 'NO'}")

    print(f"\nLoading tracker model from {repo}...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Sam3TrackerVideoModel.from_pretrained(repo, torch_dtype=torch.bfloat16).to(device)
        print(f"  tracker model class: {type(model).__name__}")
        print("  model.forward signature:")
        _sig(model.forward, "model.forward")
        # propagate_in_video_iterator is the method we rely on.
        if hasattr(model, "propagate_in_video_iterator"):
            _sig(model.propagate_in_video_iterator, "model.propagate_in_video_iterator")
        else:
            print(
                "  model.propagate_in_video_iterator: MISSING — will need another propagation path"
            )
    except Exception as e:
        print(f"FAIL loading tracker model: {type(e).__name__}: {e}")
        traceback.print_exc(limit=3)
        return None

    return {"model": model, "processor": proc, "device": device}


def probe_rvm() -> Any:
    """Load RVM via torch.hub, confirm the forward signature."""
    _rule("RVM — recurrent video matting")
    print("Loading RVM mobilenetv3 via torch.hub.load('PeterL1n/RobustVideoMatting', ...)...")
    try:
        model = torch.hub.load("PeterL1n/RobustVideoMatting", "mobilenetv3", trust_repo=True)
    except Exception as e:
        print(f"FAIL loading RVM: {type(e).__name__}: {e}")
        traceback.print_exc(limit=3)
        return None
    print(f"  model class: {type(model).__name__}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = model.to(device, dtype=dtype).eval()
    print(f"  device: {device}, dtype: {dtype}")

    print("\nForward signature:")
    _sig(model.forward, "model.forward")

    print("\nRunning one forward on a (1, 3, 256, 384) tensor with downsample_ratio=0.25...")
    # rec=[None]*4 per the README's recurrent-state contract.
    src = torch.rand(1, 3, 256, 384, device=device, dtype=dtype)
    rec: list[Any] = [None] * 4
    with torch.no_grad():
        out = model(src, *rec, downsample_ratio=0.25)
    print(f"  output type: {type(out).__name__}")
    print(f"  output len: {len(out) if hasattr(out, '__len__') else '?'}")
    if hasattr(out, "__len__"):
        for i, t in enumerate(out):
            shape = getattr(t, "shape", None)
            dtype_t = getattr(t, "dtype", None)
            print(f"  out[{i}]: shape={shape}, dtype={dtype_t}")
    return {"model": model, "device": device, "dtype": dtype}


def main() -> int:
    print("=" * 72)
    print("MATTE API PRE-FLIGHT PROBE")
    print("=" * 72)
    print(f"torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"numpy: {np.__version__}")
    try:
        import transformers

        print(f"transformers: {transformers.__version__}")
    except ImportError:
        print("transformers: NOT INSTALLED")
        return 1

    # Run each probe independently; one failure shouldn't stop the others.
    sam3_det = None
    sam3_trk = None
    rvm = None
    try:
        sam3_det = probe_sam3_detector()
    except Exception as e:
        print(f"\nSAM3 detector probe crashed: {type(e).__name__}: {e}")
        traceback.print_exc(limit=5)

    try:
        sam3_trk = probe_sam3_tracker()
    except Exception as e:
        print(f"\nSAM3 tracker probe crashed: {type(e).__name__}: {e}")
        traceback.print_exc(limit=5)

    try:
        rvm = probe_rvm()
    except Exception as e:
        print(f"\nRVM probe crashed: {type(e).__name__}: {e}")
        traceback.print_exc(limit=5)

    _rule("SUMMARY")
    print(f"SAM3 detector OK: {sam3_det is not None}")
    print(f"SAM3 tracker  OK: {sam3_trk is not None}")
    print(f"RVM           OK: {rvm is not None}")
    return 0 if (sam3_det and sam3_trk and rvm) else 1


if __name__ == "__main__":
    sys.exit(main())
