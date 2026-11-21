# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: BUSL-1.1 (see LICENSE)

"""Flow-warped temporal stabiliser: a symmetric window over any pass.

Post-processor, NOT a pass. The two-sided successor of `TemporalSmoother`
(the one-sided EMA). For every frame t it:

1. warps the neighbours t-k .. t+k (window 3, 5 or 7 => k = 1, 2, 3) onto
   frame t by chaining the motion passes the job already computed
   (forward flow for the future side, backward flow for the past side);
2. drops a neighbour sample where the chain is not trustworthy: a
   forward/backward round trip that does not come back (occlusion, bad
   flow) at any step of the chain, or a sample that leaves the frame;
3. drops a neighbour sample whose value is too far from the robust centre
   of the stack (the "spread" reject), so a flow error on fast motion
   cannot ghost. The frame's own value is only replaced when neighbours on
   BOTH sides agree against it (a single-frame pop); one side alone never
   overrides it;
4. combines the surviving samples: `median` (kills single-frame pops),
   `mean` (soft passes), or `blend` (centre-weighted, triangular).

Channel handling (by channel name):
- depth (`Z`, `Z_raw`, `*.Z`): stabilised in log space, so the spread is a
  ratio and the mean a geometric mean. Non-positive values are left as is.
- normals (`N.x`, `N.y`, `N.z` together): treated as one vector; the spread
  is the angle between normals, the result is renormalised to unit length.
- mattes (`matte.*`, `mask.*`, `key.*`): absolute spread, result clamped
  to 0-1.
- anything else: relative spread.

The output for frame t is computed from the UNSTABILISED neighbours (not
recursive), so one bad frame cannot propagate along the clip.

`applied_to` accepts channel names and shell-style patterns (`mask.*`).
Default `applied_to: []` = no-op: the post returns its input untouched.
"""

from __future__ import annotations

import fnmatch
import logging
import math
from typing import Any

import numpy as np

from live_action_aov.io.channels import CH_N_X, CH_N_Y, CH_N_Z
from live_action_aov.shared.optical_flow.cache import FlowCache

_log = logging.getLogger(__name__)

MODES = ("median", "mean", "blend")
_NORMAL_TRIPLET = (CH_N_X, CH_N_Y, CH_N_Z)
# Never touched even when a pattern matches them: they describe the run, not
# the image (flow, confidences, IDs).
_NEVER = ("motion.", "back.", "forward.", "backward.", "flow.", "crypto", "CryptoObject")


def channel_kind(name: str) -> str:
    """`depth`, `normal`, `matte` or `other` for one channel name."""
    if name in ("Z", "Z_raw") or name.endswith(".Z"):
        return "depth"
    if name in _NORMAL_TRIPLET:
        return "normal"
    if name.startswith(("matte.", "mask", "key.")):
        return "matte"
    return "other"


def default_stabilisable(names: list[str]) -> list[str]:
    """Channels worth stabilising among a pass's declared outputs: the image
    ones (depth, normals, mattes, shading), not confidences or raw copies."""
    out = []
    for n in names:
        if n.startswith(_NEVER) or "confidence" in n or n.endswith("_raw"):
            continue
        if channel_kind(n) != "other" or n.startswith(("ao.", "albedo.", "irradiance.")):
            out.append(n)
    return out


class TemporalStabiliser:
    """Flow-warped N-frame median / mean / blend over any channel set."""

    name = "temporal_stabilise"
    algorithm = "flow_warped_window_v1"

    DEFAULT_PARAMS: dict[str, Any] = {
        "applied_to": [],  # channel names or patterns; [] = no-op
        "window": 5,  # odd: 3, 5 or 7 (frames, centred on t)
        "mode": "median",  # median | mean | blend
        "fb_threshold_px": 1.0,  # round-trip error that rejects a warp step
        # Spread reject (how far a neighbour may sit from the robust centre):
        "spread_depth": 0.15,  # log units (~15 % depth ratio)
        "spread_normal_deg": 20.0,  # degrees
        "spread_matte": 0.25,  # alpha units
        "spread_other": 0.15,  # relative to |value|
        "device": "auto",  # auto | cpu | cuda
    }

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params: dict[str, Any] = dict(self.DEFAULT_PARAMS)
        if params:
            self.params.update(params)
        self.stats: dict[str, Any] = {}

    # ------------------------------------------------------------------
    def apply(
        self,
        per_frame_channels: dict[int, dict[str, np.ndarray]],
        flow_cache: FlowCache,
        shot_id: str,
        **_kwargs: Any,
    ) -> dict[int, dict[str, np.ndarray]]:
        """Return a new per-frame dict with the selected channels stabilised.

        The input dict is not mutated.
        """
        patterns: list[str] = list(self.params.get("applied_to") or [])
        if not patterns:
            return per_frame_channels
        window = int(self.params["window"])
        if window < 3 or window % 2 == 0:
            raise ValueError(f"temporal_stabilise: window must be odd and >= 3, got {window}")
        mode = str(self.params["mode"])
        if mode not in MODES:
            raise ValueError(f"temporal_stabilise: mode must be one of {MODES}, got {mode!r}")
        radius = window // 2

        import torch

        dev = str(self.params.get("device", "auto"))
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(dev)

        out: dict[int, dict[str, np.ndarray]] = {
            f: dict(channels) for f, channels in per_frame_channels.items()
        }
        frames = sorted(per_frame_channels)
        frame_set = set(frames)
        replaced = 0
        total = 0
        for t in frames:
            names = _expand(patterns, per_frame_channels[t])
            if not names:
                continue
            groups = _groups(names)
            shape = per_frame_channels[t][names[0]].shape
            chains = _build_chains(
                flow_cache, shot_id, t, radius, frame_set, shape, self.params, device
            )
            if not chains:
                continue
            for group in groups:
                res = _stabilise_group(
                    group, t, chains, per_frame_channels, mode, radius, self.params, device
                )
                if res is None:
                    continue
                values, n_rep, n_px = res
                for ch, arr in zip(group, values, strict=True):
                    out[t][ch] = arr
                replaced += n_rep
                total += n_px
        self.stats = {"pixels": total, "centre_replaced": replaced}
        if total:
            _log.info(
                "temporal_stabilise (%s, %d frames) on %s: %.2f%% of pixels were pops "
                "and got replaced.",
                mode,
                window,
                ",".join(patterns),
                100.0 * replaced / total,
            )
        return out


# ----------------------------------------------------------------------
# Channel selection
# ----------------------------------------------------------------------


def _expand(patterns: list[str], channels: dict[str, np.ndarray]) -> list[str]:
    names: list[str] = []
    for p in patterns:
        hits = [p] if p in channels else sorted(fnmatch.filter(channels.keys(), p))
        for h in hits:
            if h not in names and not h.startswith(_NEVER) and "confidence" not in h:
                names.append(h)
    return names


def _groups(names: list[str]) -> list[tuple[str, ...]]:
    """One group per channel, except a complete normal triplet (one vector)."""
    has_triplet = all(c in names for c in _NORMAL_TRIPLET)
    groups: list[tuple[str, ...]] = []
    if has_triplet:
        groups.append(_NORMAL_TRIPLET)
    for n in names:
        if has_triplet and n in _NORMAL_TRIPLET:
            continue
        groups.append((n,))
    return groups


# ----------------------------------------------------------------------
# Warp chains
# ----------------------------------------------------------------------


def _flow(
    cache: FlowCache, shot_id: str, f: int, direction: str, shape: tuple[int, ...], device: Any
) -> Any:
    """(2, H, W) torch flow at `shape`, rescaled if the cache holds another size."""
    import torch

    arr = cache.get(shot_id, f, direction)  # type: ignore[arg-type]
    if arr is None:
        return None
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).to(device)
    h, w = int(shape[0]), int(shape[1])
    if t.shape[1:] != (h, w):
        sy, sx = h / t.shape[1], w / t.shape[2]
        t = torch.nn.functional.interpolate(
            t[None], size=(h, w), mode="bilinear", align_corners=False
        )[0]
        t = torch.stack([t[0] * sx, t[1] * sy])
    return t


def _sample(img: Any, pos: Any) -> Any:
    """Bilinear sample of `img` (N, C, H, W) at pixel positions `pos` (N, H, W, 2)."""
    import torch

    h, w = img.shape[-2:]
    grid = torch.stack(
        [pos[..., 0] / max(w - 1, 1) * 2 - 1, pos[..., 1] / max(h - 1, 1) * 2 - 1], dim=-1
    )
    return torch.nn.functional.grid_sample(
        img, grid, mode="bilinear", padding_mode="border", align_corners=True
    )


def _build_chains(
    cache: FlowCache,
    shot_id: str,
    t: int,
    radius: int,
    frame_set: set[int],
    shape: tuple[int, ...],
    params: dict[str, Any],
    device: Any,
) -> list[tuple[int, Any, Any]]:
    """[(offset, pos (H, W, 2), valid (H, W) bool)] for every reachable neighbour.

    The chain for t+k samples frame t+k at the position frame-t pixel p lands
    on after k forward steps (k backward steps for t-k). A step is trusted
    where the opposite flow brings it back within `fb_threshold_px`.
    """
    import torch

    thr = float(params["fb_threshold_px"])
    h, w = int(shape[0]), int(shape[1])
    yy, xx = torch.meshgrid(
        torch.arange(h, dtype=torch.float32, device=device),
        torch.arange(w, dtype=torch.float32, device=device),
        indexing="ij",
    )
    base = torch.stack([xx, yy], dim=-1)
    chains: list[tuple[int, Any, Any]] = []
    for sign, step_dir, back_dir in ((-1, "backward", "forward"), (1, "forward", "backward")):
        pos = base.clone()
        valid = torch.ones((h, w), dtype=torch.bool, device=device)
        for k in range(1, radius + 1):
            src = t + sign * (k - 1)
            dst = t + sign * k
            if dst not in frame_set:
                break
            step = _flow(cache, shot_id, src, step_dir, shape, device)
            back = _flow(cache, shot_id, dst, back_dir, shape, device)
            if step is None or back is None:
                break
            f = _sample(step[None], pos[None])[0]  # (2, H, W) at the chain position
            nxt = pos + f.permute(1, 2, 0)
            b = _sample(back[None], nxt[None])[0]
            err = torch.sqrt((f[0] + b[0]) ** 2 + (f[1] + b[1]) ** 2)
            inside = (
                (nxt[..., 0] >= 0)
                & (nxt[..., 0] <= w - 1)
                & (nxt[..., 1] >= 0)
                & (nxt[..., 1] <= h - 1)
            )
            valid = valid & (err < thr) & inside
            pos = nxt
            chains.append((sign * k, pos.clone(), valid.clone()))
    return chains


# ----------------------------------------------------------------------
# Stack, reject, combine
# ----------------------------------------------------------------------


def _stabilise_group(
    group: tuple[str, ...],
    t: int,
    chains: list[tuple[int, Any, Any]],
    frames: dict[int, dict[str, np.ndarray]],
    mode: str,
    radius: int,
    params: dict[str, Any],
    device: Any,
) -> tuple[list[np.ndarray], int, int] | None:
    import torch

    kind = "normal" if len(group) == 3 else channel_kind(group[0])
    centre_np = [frames[t][c] for c in group]
    shape = centre_np[0].shape
    if any(a.shape != shape for a in centre_np) or len(shape) != 2:
        return None
    usable = [
        (off, pos, val)
        for off, pos, val in chains
        if all(c in frames.get(t + off, {}) and frames[t + off][c].shape == shape for c in group)
    ]
    if not usable:
        return None

    def to_space(arrs: list[np.ndarray]) -> Any:
        x = torch.from_numpy(np.stack(arrs).astype(np.float32, copy=False)).to(device)
        if kind == "depth":
            x = torch.where(x > 0, torch.log(torch.clamp(x, min=1e-12)), torch.nan)
        return torch.where(torch.isfinite(x), x, torch.nan)

    centre = to_space(centre_np)  # (C, H, W)
    neigh = torch.stack([to_space([frames[t + off][c] for c in group]) for off, _, _ in usable])
    pos = torch.stack([p for _, p, _ in usable])
    valid = torch.stack([v for _, _, v in usable])
    # NaN-aware sampling: sample the values with NaN zeroed plus a finite
    # mask, and reject a sample that touched any non-finite texel.
    fin = torch.isfinite(neigh).all(dim=1, keepdim=True).float()
    warped = _sample(torch.nan_to_num(neigh, nan=0.0), pos)
    touched = _sample(fin, pos)[:, 0] > 0.999
    valid = valid & touched
    warped = torch.where(valid[:, None], warped, torch.nan)

    stack = torch.cat([centre[None], warped], dim=0)  # (K, C, H, W), K = 1 + neighbours
    offsets = [0] + [off for off, _, _ in usable]
    tol = _tolerance(kind, params)
    ref0 = _robust_centre(stack, kind)
    d_all = _distance(stack, ref0, kind)  # (K, H, W)
    ok = torch.isfinite(d_all) & (d_all <= _tol_map(tol, ref0, kind))
    side = torch.tensor(offsets, device=device)
    past = (ok & (side < 0)[:, None, None]).any(dim=0)
    future = (ok & (side > 0)[:, None, None]).any(dim=0)
    centre_ok = ok[0]
    centre_fin = torch.isfinite(stack[0]).all(dim=0)
    use_med = centre_ok | (past & future)
    ref = torch.where(use_med[None], ref0, stack[0])
    d_ref = _distance(stack, ref, kind)
    keep = torch.isfinite(d_ref) & (d_ref <= _tol_map(tol, ref, kind))
    keep[0] = keep[0] | ~use_med  # the frame's own value always counts when it is the reference
    keep = keep & torch.isfinite(stack).all(dim=1)

    if mode == "median":
        res = _median(stack, keep, kind)
    else:
        if mode == "mean":
            wts = torch.ones(len(offsets), device=device)
        else:  # blend: triangular weights, the frame itself heaviest
            wts = torch.tensor([radius + 1 - abs(o) for o in offsets], device=device).float()
        wk = keep.float() * wts[:, None, None]
        num = (torch.nan_to_num(stack, nan=0.0) * wk[:, None]).sum(dim=0)
        den = wk.sum(dim=0).clamp(min=1e-12)
        res = num / den[None]
    # Where the frame's own value is not usable (sky in log space, NaN), keep it.
    res = torch.where(centre_fin[None] & keep.any(dim=0)[None], res, stack[0])
    n_rep = int((centre_fin & ~keep[0]).sum().item())

    if kind == "depth":
        res = torch.exp(res)
    elif kind == "normal":
        res = res / torch.linalg.vector_norm(res, dim=0, keepdim=True).clamp(min=1e-6)
    elif kind == "matte":
        res = res.clamp(0.0, 1.0)
    res_np = res.cpu().numpy().astype(np.float32, copy=False)
    outs = []
    for i, raw in enumerate(centre_np):
        r = res_np[i]
        if kind == "depth":
            # Non-positive / non-finite depth (sky codes, holes) passes through.
            r = np.where(np.isfinite(raw) & (raw > 0), r, raw)
        else:
            r = np.where(np.isfinite(raw), r, raw)
        outs.append(r.astype(np.float32, copy=False))
    return outs, n_rep, int(shape[0] * shape[1])


def _tolerance(kind: str, params: dict[str, Any]) -> float:
    if kind == "depth":
        return float(params["spread_depth"])
    if kind == "normal":
        return math.radians(float(params["spread_normal_deg"]))
    if kind == "matte":
        return float(params["spread_matte"])
    return float(params["spread_other"])


def _tol_map(tol: float, ref: Any, kind: str) -> Any:
    import torch

    if kind == "other":
        return tol * torch.clamp(ref.abs()[0], min=1e-3)
    return torch.full_like(ref[0], tol)


def _distance(stack: Any, ref: Any, kind: str) -> Any:
    """(K, H, W) distance of every sample to `ref` (C, H, W)."""
    import torch

    if kind == "normal":
        a = stack / torch.linalg.vector_norm(stack, dim=1, keepdim=True).clamp(min=1e-6)
        b = ref / torch.linalg.vector_norm(ref, dim=0, keepdim=True).clamp(min=1e-6)
        cos = (a * b[None]).sum(dim=1).clamp(-1.0, 1.0)
        return torch.acos(cos)
    return (stack - ref[None]).abs().amax(dim=1)


def _robust_centre(stack: Any, kind: str) -> Any:
    """Per-pixel, per-component median of the finite samples (C, H, W)."""
    import torch

    return _median(stack, torch.isfinite(stack).all(dim=1), kind)


def _median(stack: Any, keep: Any, kind: str) -> Any:
    """Median of the kept samples per pixel and component; the mean of the two
    middle values for an even count. (C, H, W); NaN where nothing is kept."""
    import torch

    k = keep[:, None].expand_as(stack)
    x = torch.where(k, stack, torch.inf)
    x, _ = torch.sort(x, dim=0)
    n = k.sum(dim=0)  # (C, H, W)
    lo = ((n - 1).clamp(min=0) // 2)[None]
    hi = (n // 2).clamp(max=stack.shape[0] - 1)[None]
    a = torch.gather(x, 0, lo)[0]
    b = torch.gather(x, 0, hi)[0]
    med = 0.5 * (a + b)
    med = torch.where(n > 0, med, torch.nan)
    if kind == "normal":
        med = med / torch.linalg.vector_norm(med, dim=0, keepdim=True).clamp(min=1e-6)
    return med


__all__ = ["MODES", "TemporalStabiliser", "channel_kind", "default_stabilisable"]
