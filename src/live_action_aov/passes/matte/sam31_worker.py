# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: BUSL-1.1 (see LICENSE)

"""SAM 3.1 Object Multiplex worker - runs INSIDE the SAM 3.1 environment.

Meta's SAM 3.1 needs Python 3.12 and its own torch / sam3 package, which
cannot live in LiveActionAOV's environment, so `sam31_bridge` runs this file
with that environment's interpreter. It imports nothing from LiveActionAOV
(standard library, numpy, torch, huggingface_hub and Meta's `sam3` only).

For each text concept: one session on the JPEG folder, the concept prompted
on the seed frame, masks propagated both ways. Writes one
`obj_<concept index>_<object id>.npz` (`mask`: (T, H, W) uint8) per object
and `result.json` into the output folder.

Two Windows shims, applied here instead of patching Meta's repo:
- `sam3.model.decoder` forces the FLASH_ATTENTION SDPA kernel, which the
  Windows torch wheels do not ship ("No available kernel"): it may now fall
  back to the memory-efficient and maths kernels (same result, slower).
- `start_session` passes `offload_state_to_cpu`, which the multiplex model's
  `init_state` does not accept (upstream f1647e8): unknown keywords dropped.

Usage: python sam31_worker.py --frames DIR --out DIR --concepts '["person"]'
       --seed N [--checkpoint FILE] [--repo facebook/sam3.1]
"""

from __future__ import annotations

import argparse
import glob
import inspect
import json
import os
import sys
import time


def _shims() -> None:
    import sam3.model.decoder as dec
    from torch.nn.attention import SDPBackend, sdpa_kernel

    backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
    dec.sdpa_kernel = lambda *_a, **_k: sdpa_kernel(backends)


def _drop_unknown_kwargs(pred) -> None:  # type: ignore[no-untyped-def]
    init = pred.model.init_state
    ok = set(inspect.signature(init).parameters)
    pred.model.init_state = lambda **kw: init(**{k: v for k, v in kw.items() if k in ok})


def _collect(pred, sid: str, direction: str, masks: dict) -> None:  # type: ignore[no-untyped-def]
    """Propagate one session and store each object's mask per frame."""
    import numpy as np

    req = {"type": "propagate_in_video", "session_id": sid, "propagation_direction": direction}
    for resp in pred.handle_stream_request(req):
        fi, o = resp["frame_index"], resp["outputs"]
        for oid, m in zip(o["out_obj_ids"], o["out_binary_masks"], strict=True):
            masks.setdefault(int(oid), {})[int(fi)] = np.asarray(m, dtype=np.uint8)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--concepts", required=True, help="JSON list of text prompts")
    ap.add_argument("--seed", type=int, required=True, help="seed frame index (0-based)")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--repo", default="facebook/sam3.1")
    a = ap.parse_args(argv)

    import numpy as np
    import torch
    from sam3.model_builder import build_sam3_multiplex_video_predictor

    _shims()
    n = len(glob.glob(os.path.join(a.frames, "*.jpg")))
    if n == 0:
        print(f"no JPEG frames in {a.frames}", file=sys.stderr)
        return 2
    ckpt = a.checkpoint
    if not ckpt:
        from huggingface_hub import hf_hub_download

        ckpt = hf_hub_download(a.repo, "sam3.1_multiplex.pt")
    t0 = time.time()
    pred = build_sam3_multiplex_video_predictor(checkpoint_path=ckpt, use_fa3=False)
    _drop_unknown_kwargs(pred)
    print(f"SAM 3.1 built in {time.time() - t0:.0f}s; {n} frames, seed {a.seed}", flush=True)

    os.makedirs(a.out, exist_ok=True)
    objects = []
    for ci, concept in enumerate(json.loads(a.concepts)):
        t1 = time.time()
        sid = pred.handle_request({"type": "start_session", "resource_path": a.frames})[
            "session_id"
        ]
        r = pred.handle_request(
            {"type": "add_prompt", "session_id": sid, "frame_index": a.seed, "text": concept}
        )
        seeds = [int(x) for x in r["outputs"]["out_obj_ids"]]
        masks: dict[int, dict[int, np.ndarray]] = {}

        try:
            _collect(pred, sid, "both", masks)
        except TypeError:
            _collect(pred, sid, "forward", masks)
            _collect(pred, sid, "backward", masks)
        pred.handle_request({"type": "close_session", "session_id": sid})
        for oid, fm in sorted(masks.items()):
            h, w = next(iter(fm.values())).shape
            stack = np.zeros((n, h, w), np.uint8)
            for fi, m in fm.items():
                if 0 <= fi < n:
                    stack[fi] = m
            name = f"obj_{ci:02d}_{oid:04d}.npz"
            np.savez_compressed(os.path.join(a.out, name), mask=stack)
            objects.append({"file": name, "concept": concept, "obj_id": oid})
        print(
            f"'{concept}': {len(seeds)} seeded, {len(masks)} tracked in {time.time() - t1:.0f}s",
            flush=True,
        )
    peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    with open(os.path.join(a.out, "result.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "frames": n,
                "seed": a.seed,
                "objects": objects,
                "seconds": round(time.time() - t0),
                "peak_vram_gb": round(peak, 1),
            },
            f,
            indent=1,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
