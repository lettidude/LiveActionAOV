# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: MIT

"""Screen pull — classic whole-plate chroma key (premult RGBA + despill).

This is the KEYER product, distinct from the SAM-bounded `chroma_key`
refiner (which produces per-character rotos):

    screen_pull  ->  key.rgba : the whole plate keyed against the screen.
                     key.a = 1 everywhere that is not screen, 0 on the
                     screen, soft in between (hair, motion blur);
                     key.rgb = DESPILLED plate premultiplied by key.a.
                     A comper drops it straight over a background.
    chroma_key   ->  per-object roto (SAM decides WHO, key does the edge).

No SAM, no models, no weights — pure colour-difference maths, standalone
(`requires_artifacts = []`), streams frame by frame (no full-clip stack,
memory-flat on 1000-frame clips).

Screen colour is auto-detected per shot (dominant green-vs-blue
colour-difference population); the pull is auto-calibrated per frame
against the measured screen so uneven screens map pure screen to alpha 0.

The pass keys on SCENE-LINEAR pixels (`input_colorspace =
"scene_linear"` — the executor hands it the OCIO-linearized plate,
skipping exposure/AgX/EOTF). Keying on the display-transformed plate
was the 2026-08-12 field failure: AgX desaturation compressed the
green screen and green-ish plate content (foliage, glossy-car
reflections) toward the same pale values, so the calibrated pull keyed
half the plate. In scene-linear the screen stays strongly chromatic
and the discriminant holds.

Calibration is two-stage per frame: screen candidates = pixels with
d > 0.5 * p99(d) (works even when the screen is a small fraction of
frame — a whole-frame p90 needs the screen to cover >10%), d_ref =
their median. A per-shot floor (3/4 of the first calibrated d_ref) stops
frames where the screen leaves view from collapsing d_ref onto
foliage. key.rgb stays unclamped scene-linear (premultiplied) so comp
keeps the plate's dynamic range.

Despill (green example): g' = min(g, blend of max(r,b) and (r+b)/2) —
kills the green bounce on skin/edges before premultiplying.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from live_action_aov.core.pass_base import (
    ChannelSpec,
    License,
    PassType,
    TemporalMode,
    UtilityPass,
)
from live_action_aov.io.channels import CH_KEY_A, CH_KEY_B, CH_KEY_G, CH_KEY_R

_log = logging.getLogger(__name__)


class ScreenPullPass(UtilityPass):
    name = "screen_pull"
    version = "0.1.0"
    license = License(
        spdx="MIT",
        commercial_use=True,
        commercial_tool_resale=True,
        notes=(
            "Pure colour-difference keying + despill (no model, no weights) "
            "— commercial-safe with no provenance caveats."
        ),
    )
    pass_type = PassType.SEMANTIC
    temporal_mode = TemporalMode.VIDEO_CLIP  # run_shot streams frame-by-frame
    # SCENE-LINEAR input — see module docstring. The executor recognises
    # this value and hands run_shot a LinearizedReader.
    input_colorspace = "scene_linear"

    produces_channels = [
        ChannelSpec(name=CH_KEY_R, description="Keyed plate R (despilled, premultiplied)"),
        ChannelSpec(name=CH_KEY_G, description="Keyed plate G (despilled, premultiplied)"),
        ChannelSpec(name=CH_KEY_B, description="Keyed plate B (despilled, premultiplied)"),
        ChannelSpec(name=CH_KEY_A, description="Screen-pull alpha (screen=0, subject=1)"),
    ]

    requires_artifacts: list[str] = []
    provides_artifacts: list[str] = []
    smoothable_channels: list[str] = []  # deterministic per-frame maths

    DEFAULT_PARAMS: dict[str, Any] = {
        "screen": "auto",  # "auto" | "green" | "blue"
        # alpha = 1 - clip((d / d_ref) * gain - lift). d_ref measured per
        # frame from the screen population; gain multiplies the calibrated
        # pull (1.0 = calibrated), lift eats screen noise.
        "key_gain": 1.0,
        "key_lift": 0.02,
        # Despill: 0 = off; 1 = full clamp of the screen primary to the
        # despill reference. Reference blends max(other) and mean(other).
        "despill": 1.0,
        "despill_balance": 0.5,  # 0 = max(r,b) (conservative), 1 = (r+b)/2
    }

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        for k, v in self.DEFAULT_PARAMS.items():
            self.params.setdefault(k, v)
        self._model: Any = None
        # First successfully-calibrated d_ref of the shot — later frames
        # never calibrate below 3/4 of it (screen leaving frame must not
        # re-target the pull onto foliage), and it damps flicker.
        self._shot_d_ref: float | None = None

    # No weights — sentinel loader keeps executor lifecycle happy.
    def _load_model(self) -> None:
        self._model = object()

    # ------------------------------------------------------------------
    # Keying maths
    # ------------------------------------------------------------------

    @staticmethod
    def _colour_diff(rgb: np.ndarray, screen: str) -> np.ndarray:
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        if screen == "blue":
            return b - np.maximum(r, g)
        return g - np.maximum(r, b)

    @classmethod
    def _detect_screen(cls, rgb: np.ndarray) -> str:
        """Dominant screen primary from the whole frame: compare the deep
        (p90) green-difference vs blue-difference populations."""
        dg = float(np.percentile(cls._colour_diff(rgb, "green"), 90))
        db = float(np.percentile(cls._colour_diff(rgb, "blue"), 90))
        if max(dg, db) < 0.05:
            _log.warning(
                "screen_pull: no strong green/blue screen found in frame "
                "(p90 diff g=%.3f b=%.3f) — the pull will be weak. Is this "
                "actually a screen plate?",
                dg,
                db,
            )
        return "blue" if db > dg else "green"

    def _pull_alpha(self, rgb: np.ndarray, screen: str) -> np.ndarray:
        d = self._colour_diff(rgb, screen)
        # Two-stage calibration: the screen population is "pixels close to
        # the frame's deepest colour-difference" (p99 top). Median of that
        # population = d_ref. Robust to small screens (2% of frame is
        # plenty) where a whole-frame percentile lands on foliage instead.
        d99 = float(np.percentile(d, 99))
        if d99 >= 0.02:
            cand = d[d > 0.5 * d99]
            d_ref = max(float(np.median(cand)) * 0.9, 1e-4)
        else:
            # No measurable screen this frame — weak fixed pull rather
            # than amplifying noise.
            d_ref = 0.45
        if self._shot_d_ref is None:
            if d99 >= 0.02:
                self._shot_d_ref = d_ref
        else:
            # Per-frame calibration only tracks mild screen/lighting drift;
            # a big drop means the screen left frame and the top-percentile
            # population is now plate content — don't re-target onto it.
            d_ref = max(d_ref, 0.75 * self._shot_d_ref)
        gain = float(self.params.get("key_gain", 1.0))
        lift = float(self.params.get("key_lift", 0.02))
        return (1.0 - np.clip((d / d_ref) * gain - lift, 0.0, 1.0)).astype(np.float32)

    def _despill(self, rgb: np.ndarray, screen: str) -> np.ndarray:
        amount = float(np.clip(self.params.get("despill", 1.0), 0.0, 1.0))
        if amount <= 0.0:
            return rgb
        bal = float(np.clip(self.params.get("despill_balance", 0.5), 0.0, 1.0))
        out = rgb.copy()
        if screen == "blue":
            others_max = np.maximum(rgb[..., 0], rgb[..., 1])
            others_mean = 0.5 * (rgb[..., 0] + rgb[..., 1])
            ch = 2
        else:
            others_max = np.maximum(rgb[..., 0], rgb[..., 2])
            others_mean = 0.5 * (rgb[..., 0] + rgb[..., 2])
            ch = 1
        limit = (1.0 - bal) * others_max + bal * others_mean
        spilled = out[..., ch] > limit
        out[..., ch] = np.where(
            spilled, out[..., ch] + amount * (limit - out[..., ch]), out[..., ch]
        )
        return out

    # ------------------------------------------------------------------
    # Shot-level: stream frames, no full-clip stack
    # ------------------------------------------------------------------

    def run_shot(
        self,
        reader: Any,
        frame_range: tuple[int, int],
    ) -> dict[int, dict[str, np.ndarray]]:
        first, last = frame_range
        screen_param = str(self.params.get("screen", "auto")).lower()
        screen = screen_param if screen_param in ("green", "blue") else ""

        out: dict[int, dict[str, np.ndarray]] = {}
        for f in range(first, last + 1):
            rgb = np.asarray(reader.read_frame(f)[0], dtype=np.float32)[..., :3]
            if not screen:
                screen = self._detect_screen(rgb)
                _log.info("screen_pull: detected %s screen", screen)
            alpha = self._pull_alpha(rgb, screen)
            # No [0,1] clip: the plate is scene-linear and key.rgb should
            # keep its dynamic range (negatives still floored — they are
            # sensor/decode noise, not signal).
            despilled = self._despill(np.maximum(rgb, 0.0), screen)
            premult = despilled * alpha[..., None]
            out[f] = {
                CH_KEY_R: premult[..., 0].astype(np.float32),
                CH_KEY_G: premult[..., 1].astype(np.float32),
                CH_KEY_B: premult[..., 2].astype(np.float32),
                CH_KEY_A: alpha,
            }
        return out

    # Per-frame lifecycle unused — run_shot is the entry.
    def preprocess(self, frames: np.ndarray) -> Any:
        return frames

    def infer(self, tensor: Any) -> Any:
        return tensor

    def postprocess(self, tensor: Any) -> dict[str, np.ndarray]:
        return {}


__all__ = ["ScreenPullPass"]
