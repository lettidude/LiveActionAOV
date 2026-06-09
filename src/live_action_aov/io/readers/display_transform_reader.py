# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: MIT

"""Reader wrapper that applies a clip-uniform display transform on read.

Background
----------

Depth / normals / matte models (DA-v2, DSINE, SAM 3, RVM) are trained on
sRGB-display images. Scene-referred linear plates (ACES, lin_rec709,
ACEScg) look "washed out" to these models — peak values sit near the
noise floor of the model's training distribution, so the network falls
back to 2D priors (depth collapses to a y-axis gradient; normals jitter
frame-to-frame on low-signal input).

This wrapper fixes that by applying a display transform BEFORE the pass
sees a frame:

    linearize (OCIO) → clip-wide auto-exposure → AgX tonemap → sRGB EOTF → [0,1]

The "clip-wide" part is critical — per-frame exposure would introduce
flicker on moving-camera shots (design §7 trap 4). Exposure is sampled
on N evenly-spaced frames at `analyze` time and used for all subsequent
`read_frame` calls.

Why a wrapper and not a mutation in each pass
---------------------------------------------

Every pass called `reader.read_frame(f)` then did
`np.clip(frames, 0.0, 1.0)` with raw linear input. Hard-coding the
transform into each pass would mean four copies to keep in sync, and
would make unit tests (which feed synthetic [0,1] arrays directly)
awkward. Wrapping the reader keeps the transform in one place and is
off-by-default — set `Shot.apply_display_transform=True` in the
executor to turn it on.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from live_action_aov.core.pass_base import DisplayTransformParams
from live_action_aov.io import ocio_color
from live_action_aov.io.colorspace_detect import detect_colorspace
from live_action_aov.io.display_transform import DisplayTransform
from live_action_aov.io.readers.base import ImageSequenceReader

_log = logging.getLogger(__name__)


def _sanitize_nonfinite(frames: np.ndarray, *, frame: int, where: str) -> np.ndarray:
    """Replace NaN / ±Inf pixels with 0.0 and log how many were found.

    This is the pipeline's defence against corrupt input. Comp-work
    plates routinely carry non-finite pixels (unpremult divides, bad
    merges, expression nodes); some renderers emit them too. Such a
    pixel is poison: ``np.clip`` does NOT strip NaN/Inf, exposure is
    typically a no-op (manual EV 0), and AgX's internal clip only bounds
    *finite* values — so a single NaN sails through every transform into
    the model, which then emits NaN into EVERY sidecar channel. Enough
    of them silently turn a whole batch to garbage (or a blank matte,
    since SAM 3 detects nothing on NaN input).

    Replacing with 0.0 (not a large finite value) keeps the fix neutral:
    a corrupt pixel reads as "empty/black" rather than a blown highlight.
    The WARNING is the diagnostic — if it fires on the *source* stage the
    plate is the culprit; if only on *post-transform* the colour decode
    produced the non-finite values.
    """
    finite = np.isfinite(frames)
    bad = int(frames.size) - int(finite.sum())
    if bad:
        _log.warning(
            "frame %d: %d non-finite pixel value(s) (NaN/Inf) in %s — "
            "sanitized to 0.0. Source plate likely contains bad pixels "
            "(unpremult divide / expression / render hole); clean it "
            "upstream to avoid blank or NaN passes.",
            frame,
            bad,
            where,
        )
        return np.nan_to_num(frames, nan=0.0, posinf=0.0, neginf=0.0)
    return frames


class DisplayTransformedReader(ImageSequenceReader):
    """Wrap a base reader, apply a clip-uniform display transform on read.

    After construction, call `analyze(frame_range)` once to sample the
    clip and compute the exposure EV. `read_frame` then returns display-
    space pixels (sRGB/rec709/linear per params, clamped [0,1]).

    Pass attrs are forwarded unchanged — only pixel values are transformed.
    """

    def __init__(
        self,
        base: ImageSequenceReader,
        params: DisplayTransformParams,
        colorspace_override: str | None = None,
    ) -> None:
        # ImageSequenceReader's __init__ wants folder + pattern; we just
        # defer everything to the base so subclassing stays cheap.
        self._base = base
        self.folder = base.folder
        self.sequence_pattern = base.sequence_pattern
        self._params = params
        self._colorspace_override = colorspace_override
        self._transform = DisplayTransform()
        self._analysis: dict[str, Any] | None = None
        self._working_space = "acescg"
        # Warn at most once per shot if OCIO can't decode the source
        # colorspace — otherwise it's one line per frame (50+ for a clip).
        self._linearize_warned = False

    # --- Analysis ------------------------------------------------------

    def analyze(self, frame_range: tuple[int, int]) -> dict[str, Any]:
        """Sample up to `params.sample_frames` evenly-spaced frames, run
        `DisplayTransform.analyze_clip`, and cache the result. Must be
        called before `read_frame`.
        """
        first, last = frame_range
        n = max(1, int(self._params.sample_frames))
        if last > first:
            step = max(1, (last - first) // max(1, n - 1))
            sampled_indices = list(range(first, last + 1, step))[:n]
        else:
            sampled_indices = [first]
        sample_linear: list[np.ndarray] = []
        colorspace = self._resolve_colorspace()
        for fi in sampled_indices:
            raw, _ = self._base.read_frame(fi)
            # Sanitize before exposure analysis: a single Inf would
            # poison the percentile/median and yield a garbage clip EV.
            raw = _sanitize_nonfinite(raw, frame=fi, where="source plate (exposure sample)")
            lin = self._linearize(raw, colorspace)
            sample_linear.append(lin)
        analysis = self._transform.analyze_clip(
            sample_linear, self._params, working_space=self._working_space
        )
        self._analysis = analysis
        self._params.computed_exposure_ev = float(analysis.get("ev", 0.0))
        return analysis

    # --- ImageSequenceReader API (delegate where possible) -------------

    def frame_range(self) -> tuple[int, int]:
        return self._base.frame_range()

    def resolution(self) -> tuple[int, int]:
        return self._base.resolution()

    def pixel_aspect(self) -> float:
        return self._base.pixel_aspect()

    def read_frame(self, frame: int) -> tuple[np.ndarray, dict[str, Any]]:
        raw, attrs = self._base.read_frame(frame)
        # Source guard: strip NaN/Inf the plate carries before they can
        # propagate through linearize → transform → model → sidecar.
        # This is the primary fix; the WARNING here means the plate is
        # the culprit.
        raw = _sanitize_nonfinite(raw, frame=frame, where="source plate")
        if self._analysis is None:
            # Caller forgot to analyze — behave as a passthrough rather
            # than silently drop exposure. Most likely a test path.
            return raw, attrs
        colorspace = self._resolve_colorspace(attrs)
        lin = self._linearize(raw, colorspace)
        out = self._transform.apply(lin, self._params, self._analysis)
        # Defensive net: a colour decode (e.g. an out-of-domain Log
        # inverse) can manufacture non-finite values even from a clean
        # plate. A WARNING here (with no source-stage warning) points at
        # the transform rather than the plate.
        out = _sanitize_nonfinite(out, frame=frame, where="post-transform output")
        return out.astype(np.float32, copy=False), attrs

    # --- Helpers -------------------------------------------------------

    def _resolve_colorspace(self, attrs: dict[str, Any] | None = None) -> str:
        if self._colorspace_override:
            return self._colorspace_override
        if attrs is None:
            # Peek the first frame's attrs.
            first = self._base.frame_range()[0]
            _, attrs = self._base.read_frame(first)
        return _sniff_colorspace_extended(attrs, fallback=self._working_space)

    def _linearize(self, frames: np.ndarray, colorspace: str) -> np.ndarray:
        # Fast path: already linear → skip the OCIO round-trip.
        if colorspace in _LINEAR_COLORSPACES:
            return frames.astype(np.float32, copy=False)
        try:
            return ocio_color.to_linear(frames, colorspace)
        except Exception as exc:
            # OCIO misconfiguration shouldn't break the pipeline — fall
            # back to the narrow hard-coded set (sRGB / linear). But this
            # is a *silent correctness trap*: if the active OCIO config
            # has no colorspace named for `colorspace` (e.g. a studio
            # config missing "ARRI LogC4"), the fallback can't decode a
            # Log curve and returns the pixels UNCHANGED — the model then
            # sees Log-encoded (washed-out / near-black) input and the
            # passes come out wrong or dark. Surface it once per shot so
            # the cause is visible in the log instead of guessed at.
            if not self._linearize_warned:
                self._linearize_warned = True
                handled = colorspace in ocio_color._SRGB_NAMES or colorspace in ocio_color._LINEAR_NAMES
                if not handled:
                    _log.warning(
                        "could not linearize colorspace %r via OCIO (%s); "
                        "the fallback cannot decode this space so pixels are "
                        "passed through UNLINEARIZED — passes will look wrong "
                        "or dark. Point $OCIO at a config that defines this "
                        "colorspace, or set the shot's colorspace explicitly.",
                        colorspace,
                        exc,
                    )
            return ocio_color._fallback_to_linear(frames, colorspace)


# Colorspaces we know are already linear — skip the OCIO step for them.
# Keep this list conservative; when in doubt, go through OCIO.
_LINEAR_COLORSPACES = {
    "linear",
    "lin_rec709",
    "linear_rec709",
    "lin_srgb",
    "linear_srgb",
    "scene_linear",
    "acescg",
    "ACES - ACEScg",
    "Utility - Linear - Rec.709",
    "Linear Rec.709",
}


def _sniff_colorspace_extended(attrs: dict[str, Any], fallback: str) -> str:
    """Delegate to the full detection ladder in `colorspace_detect`.

    The detector runs camera-metadata sniffing BEFORE trusting
    `oiio:ColorSpace`, which is critical for ARRI / Sony plates whose
    OIIO stamp routinely lies about Log encoding. An older version of
    this function read the header attributes directly and short-
    circuited before the camera-metadata check, so lying-tag plates
    went through the display transform as lin_rec709 — visually wrong
    and quietly so. The detector now owns the full ladder; this
    wrapper exists only to fold `detected.detected` out of the
    structured result.
    """
    result = detect_colorspace(attrs, sample_pixels=None, fallback=fallback)
    return result.detected


__all__ = ["DisplayTransformedReader"]
