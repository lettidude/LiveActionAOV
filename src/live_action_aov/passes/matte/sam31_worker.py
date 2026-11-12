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

Small objects (the matte pass's `small_objects` on). Every flag below is
optional; left out, the run above is unchanged:
- `--max-objects N`: the model's object cap (left out: the builder's own
  default); detections over the cap are dropped.
- `--post-batch N`: frames the model post-processes at once (left out: its
  default). Its GPU memory grows with frames x objects x frame size, so a
  bigger object cap on a big plate wants a smaller batch.
- `--seeds '[50, 0, 99]'`: one session per seed frame instead of `--seed`
  (someone hidden or not yet in frame at one seed is prompted at another).
  The first seed's session on the plate is the primary one and must work;
  any other session SAM 3.1 fails on is logged in `sessions` and skipped.
- `--tiles '["dir1", ...]'`: more JPEG folders (crops of the plate), run
  with the same concepts and seeds after `--frames`; `src` in result.json
  says which folder an object came from (0 = `--frames`, 1.. = the tiles).
- `--crops`: masks stored as per-frame boxes + crops (`boxes` (T, 4) as
  y0 x0 y1 x1, `data` the crops flattened, `hw`) instead of (T, H, W)
  stacks, which grow with objects x frames x resolution.

Two Windows shims, applied here instead of patching Meta's repo:
- `sam3.model.decoder` forces the FLASH_ATTENTION SDPA kernel, which the
  Windows torch wheels do not ship ("No available kernel"): it may now fall
  back to the memory-efficient and maths kernels (same result, slower).
- `start_session` passes `offload_state_to_cpu`, which the multiplex model's
  `init_state` does not accept (upstream f1647e8): unknown keywords dropped.

Usage: python sam31_worker.py --frames DIR --out DIR --concepts '["person"]'
       --seed N [--checkpoint FILE] [--repo facebook/sam3.1]
       [--max-objects N] [--post-batch N] [--seeds JSON] [--tiles JSON] [--crops]
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


def _collect_crops(pred, sid: str, direction: str, masks: dict) -> None:  # type: ignore[no-untyped-def]
    """`_collect`, each frame kept as (y0, x0, crop, h, w); empty frames
    skipped (the object id is still recorded)."""
    import numpy as np

    req = {"type": "propagate_in_video", "session_id": sid, "propagation_direction": direction}
    for resp in pred.handle_stream_request(req):
        fi, o = resp["frame_index"], resp["outputs"]
        for oid, m in zip(o["out_obj_ids"], o["out_binary_masks"], strict=True):
            a = np.asarray(m, dtype=np.uint8)
            fm = masks.setdefault(int(oid), {})
            rows, cols = np.flatnonzero(a.any(1)), np.flatnonzero(a.any(0))
            if rows.size:
                y0, y1, x0, x1 = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
                fm[int(fi)] = (y0, x0, a[y0:y1, x0:x1].copy(), a.shape[0], a.shape[1])


def _save_crops(path: str, fm: dict, n: int) -> None:
    """One object's box crops -> npz: `boxes` (n, 4) y0 x0 y1 x1 (zeros on
    empty frames), `data` the crops flattened in frame order, `hw`."""
    import numpy as np

    boxes = np.zeros((n, 4), np.int32)
    parts, hw = [], (0, 0)
    for fi in sorted(fm):
        y0, x0, c, h, w = fm[fi]
        hw = (h, w)
        if 0 <= fi < n:
            boxes[fi] = (y0, x0, y0 + c.shape[0], x0 + c.shape[1])
            parts.append(c.reshape(-1))
    data = np.concatenate(parts) if parts else np.zeros(0, np.uint8)
    np.savez_compressed(path, boxes=boxes, data=data, hw=np.asarray(hw, np.int32))


def _run_small(pred, a, n: int) -> tuple[list, list]:  # type: ignore[no-untyped-def]
    """The small-objects run: every folder (plate, then tiles) x concept x
    seed frame is one session; masks kept as box crops. The primary
    session (the plate on `--seed`) must work; any other one that SAM 3.1
    fails on is logged in `sessions` and skipped."""
    import torch

    # Seed frames in the order given: the first one is the primary seed.
    seeds = [int(s) for s in json.loads(a.seeds)] if a.seeds else [a.seed]
    seeds = list(dict.fromkeys(min(max(s, 0), n - 1) for s in seeds))
    dirs = [a.frames, *(json.loads(a.tiles) if a.tiles else [])]
    objects, sessions = [], []
    session = -1
    for src, fdir in enumerate(dirs):
        for ci, concept in enumerate(json.loads(a.concepts)):
            for si, seed in enumerate(seeds):
                session += 1
                t1 = time.time()
                sid = pred.handle_request({"type": "start_session", "resource_path": fdir})[
                    "session_id"
                ]
                masks: dict = {}
                try:
                    r = pred.handle_request(
                        {
                            "type": "add_prompt",
                            "session_id": sid,
                            "frame_index": seed,
                            "text": concept,
                        }
                    )
                    n_seed = len(r["outputs"]["out_obj_ids"])
                    try:
                        _collect_crops(pred, sid, "both", masks)
                    except TypeError:
                        _collect_crops(pred, sid, "forward", masks)
                        _collect_crops(pred, sid, "backward", masks)
                except Exception as e:
                    if src == 0 and si == 0:
                        raise  # the primary session must work, as in the plain run
                    print(
                        f"src {src} '{concept}' seed {seed}: session SKIPPED - SAM 3.1 error: {e}",
                        flush=True,
                    )
                    sessions.append(
                        {"src": src, "concept": concept, "seed": seed, "skipped": str(e)[-500:]}
                    )
                    try:
                        pred.handle_request({"type": "close_session", "session_id": sid})
                    except Exception:
                        pass
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                pred.handle_request({"type": "close_session", "session_id": sid})
                kept = 0
                for oid, fm in sorted(masks.items()):
                    if not fm:
                        continue
                    name = f"obj_s{src:02d}_k{si:02d}_c{ci:02d}_{oid:04d}.npz"
                    _save_crops(os.path.join(a.out, name), fm, n)
                    objects.append(
                        {
                            "file": name,
                            "concept": concept,
                            "obj_id": oid,
                            "src": src,
                            "seed": seed,
                            "session": session,
                            "format": "crops",
                        }
                    )
                    kept += 1
                sessions.append(
                    {
                        "src": src,
                        "concept": concept,
                        "seed": seed,
                        "seeded": n_seed,
                        "tracked": kept,
                    }
                )
                print(
                    f"src {src} '{concept}' seed {seed}: {n_seed} seeded, {kept} tracked "
                    f"in {time.time() - t1:.0f}s",
                    flush=True,
                )
    return objects, sessions


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--concepts", required=True, help="JSON list of text prompts")
    ap.add_argument("--seed", type=int, required=True, help="seed frame index (0-based)")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--repo", default="facebook/sam3.1")
    ap.add_argument("--max-objects", type=int, default=0, help="0 = the builder's default")
    ap.add_argument("--post-batch", type=int, default=0, help="0 = the model's default")
    ap.add_argument("--seeds", default="", help="JSON list of seed frames (overrides --seed)")
    ap.add_argument("--tiles", default="", help="JSON list of more JPEG folders")
    ap.add_argument("--crops", action="store_true", help="store box crops, not stacks")
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
    build_kw = {"max_num_objects": a.max_objects} if a.max_objects > 0 else {}
    pred = build_sam3_multiplex_video_predictor(checkpoint_path=ckpt, use_fa3=False, **build_kw)
    _drop_unknown_kwargs(pred)
    if a.post_batch > 0 and hasattr(pred.model, "postprocess_batch_size"):
        pred.model.postprocess_batch_size = a.post_batch
    print(f"SAM 3.1 built in {time.time() - t0:.0f}s; {n} frames, seed {a.seed}", flush=True)

    os.makedirs(a.out, exist_ok=True)
    small = bool(a.seeds or a.tiles or a.crops)
    objects, sessions = _run_small(pred, a, n) if small else ([], [])
    for ci, concept in enumerate([] if small else json.loads(a.concepts)):
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
                **({"sessions": sessions} if small else {}),
            },
            f,
            indent=1,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
