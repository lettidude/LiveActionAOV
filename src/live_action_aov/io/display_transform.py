"""Display transform: scene-referred → model-friendly (design §7).

Algorithm (per clip, uniform across frames):

1. Linearize via OCIO (`to_linear`)
2. Clip-wide auto-exposure — sample N frames, pick percentile luminance,
   solve for single `E` (design §7 — this is non-negotiable; per-frame
   exposure causes flicker in model input — trap 4)
3. Apply exposure: frame *= 2^E
4. Tone map (AgX default; Filmic / Reinhard / none supported)
5. EOTF (sRGB / rec709 / linear)
6. Clamp [0, 1]

`analyze_clip` is heavy (reads N frames); `apply` is cheap (milliseconds) so
the GUI can call it on every slider drag.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from live_action_aov.core.pass_base import DisplayTransformParams
from live_action_aov.io import ocio_color

# Rec.709 luminance coefficients — standard for VFX display transforms.
# For ACEScg working space, use ACES AP1 luminance coefficients.
LUMA_REC709 = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
LUMA_AP1 = np.array([0.2722287168, 0.6740817658, 0.0536895174], dtype=np.float32)


class DisplayTransform:
    """Analyze + apply a display transform for one clip."""

    def analyze_clip(
        self,
        sample_frames: list[np.ndarray],
        params: DisplayTransformParams,
        working_space: str = "acescg",
    ) -> dict[str, Any]:
        """Compute the clip-wide exposure EV from a sample of frames.

        `sample_frames` should already be linearized (call `ocio_color.to_linear`
        before sampling). Returns a dict placed onto the shot so
        `apply` is cheap.
        """
        luma_coeffs = LUMA_AP1 if working_space == "acescg" else LUMA_REC709

        if params.manual_exposure_ev is not None:
            return {"ev": float(params.manual_exposure_ev), "source": "manual"}
        if not params.auto_exposure_enabled:
            return {"ev": 0.0, "source": "disabled"}

        values: list[float] = []
        for frame in sample_frames:
            if frame.ndim == 2:
                lum = frame.astype(np.float32)
            else:
                lum = frame[..., :3].astype(np.float32) @ luma_coeffs
            # Filter out true blacks (border pixels) which skew percentile.
            lum = lum[lum > 1e-6]
            if lum.size == 0:
                continue
            if params.exposure_anchor == "median":
                values.append(float(np.median(lum)))
            elif params.exposure_anchor == "p75":
                values.append(float(np.percentile(lum, 75)))
            else:  # mean_log
                values.append(float(np.exp(np.mean(np.log(lum)))))
        if not values:
            return {"ev": 0.0, "source": "no_samples"}

        clip_percentile = float(np.median(values))
        # Solve 2^E * Y = target ⇒ E = log2(target/Y)
        target = params.exposure_target
        ev = float(np.log2(max(1e-8, target) / max(1e-8, clip_percentile)))
        return {"ev": ev, "source": "auto", "sampled_luma": clip_percentile}

    def apply(
        self,
        frames: np.ndarray,
        params: DisplayTransformParams,
        analysis: dict[str, Any],
    ) -> np.ndarray:
        """Apply the display transform to linear `frames`.

        `frames` is (H, W, C) or (N, H, W, C) float32 in the linear working
        space. Returns an array of the same shape in the output EOTF space,
        optionally clamped to [0, 1].
        """
        ev = float(analysis.get("ev", 0.0))
        out = frames.astype(np.float32, copy=False) * (2.0**ev)
        out = _tonemap(out, params.tonemap)
        # AgX's sigmoid already produces display-referred values — applying an
        # sRGB EOTF on top is a double display transform and was the second
        # half of the "near-white → ImageNet-black" DA-V2 failure. For every
        # other tonemap (linear / reinhard / filmic) the output is still
        # linear-light, so the caller-requested EOTF applies.
        if str(params.tonemap).lower() != "agx":
            out = _apply_eotf(out, params.output_eotf)
        if params.clamp:
            out = np.clip(out, 0.0, 1.0)
        return out


# ---------------------------------------------------------------------------
# Tonemappers — kept inline in v1. Phase 6 can refactor into plugins behind
# the `live_action_aov.tonemappers` entry-point group.
# ---------------------------------------------------------------------------


def _tonemap(x: np.ndarray, name: str) -> np.ndarray:
    if name in ("none", "linear", None):
        return x
    if name == "reinhard":
        return x / (1.0 + x)
    if name == "filmic":
        return _hable(x)
    if name == "agx":
        return _agx(x)
    # Unknown: passthrough with silent fallback.
    return x


def _hable(x: np.ndarray) -> np.ndarray:
    """Hable (Uncharted 2) filmic curve."""
    A, B, C, D, E, F = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30
    W = 11.2
    num = x * (A * x + C * B) + D * E
    den = x * (A * x + B) + D * F
    curve = num / np.maximum(den, 1e-8) - E / F
    white_num = W * (A * W + C * B) + D * E
    white_den = W * (A * W + B) + D * F
    white_scale = 1.0 / (white_num / white_den - E / F)
    return curve * white_scale


def _agx(x: np.ndarray) -> np.ndarray:
    """AgX-lite: log encode → sigmoid → display.

    Sobotka / Minimal-AgX reference polynomial. The earlier fit in this
    file had coefficients that evaluated to a nearly flat curve near
    0.85–0.90 across the whole midtone range — the diagnostic PNG dump
    for DA-V2 showed 18% grey inputs emerging at ~0.91 (near-white), so
    the model saw a blown plate and fell back to a 2D radial prior.
    The coefficients below are the Three.js / Iolite port that round-trips
    18% grey → ~0.49 display and 1.0 → 1.0.

    Output is already **display-referred** (sigmoid bakes the display
    gamma); callers must not apply an additional sRGB EOTF on top.
    """
    # Log encode over a fixed EV range.
    x = np.clip(x, 1e-8, None)
    min_ev, max_ev = -12.47393, 4.026069
    log_x = np.log2(x)
    norm = (log_x - min_ev) / (max_ev - min_ev)
    norm = np.clip(norm, 0.0, 1.0)
    # 6th-order polynomial sigmoid (Minimal AgX default contrast).
    t = norm
    t2 = t * t
    t4 = t2 * t2
    sig = (
        15.5 * t4 * t2
        - 40.14 * t4 * t
        + 31.96 * t4
        - 6.868 * t2 * t
        + 0.4298 * t2
        + 0.1191 * t
        - 0.00232
    )
    return np.clip(sig, 0.0, 1.0)


def _apply_eotf(x: np.ndarray, eotf: str) -> np.ndarray:
    if eotf == "linear":
        return x
    if eotf == "srgb":
        return ocio_color._linear_to_srgb(x)
    if eotf == "rec709":
        # Rec.709 OETF is close enough to a pure 2.4 gamma for display work.
        return np.power(np.clip(x, 0.0, 1.0), 1.0 / 2.4)
    return x


__all__ = ["DisplayTransform", "DisplayTransformParams"]
