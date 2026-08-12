# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: MIT

"""Chroma-key matte refiner — SAM3-bounded screen pull (green OR blue).

On a green/blue screen, a colour-difference key beats any matting model at
the thing that matters for roto: hair, motion blur and semi-transparency
are resolved **physically** (the screen shows through mixed pixels), not
hallucinated. The classic weakness of a key — it pulls *everything*
non-screen (rigs, stands, other people) — is exactly what SAM 3 solves:

    SAM 3  = WHO   (per-character isolation, garbage bound)
    key    = HOW MUCH (the actual soft alpha, colour-difference)

The maths is the decades-old colour-difference pull (no ML, no weights,
no licence baggage — CPU-cheap, temporally stable by construction):

    green screen:  d = G - max(R, B)
    blue  screen:  d = B - max(R, G)
    alpha_key = 1 - clip(d * gain - lift, 0, 1)

Screen colour is auto-detected per shot by sampling the plate OUTSIDE the
SAM hard mask (the screen region) — `screen: "auto" | "green" | "blue"`
overrides it. Alpha only: no despill is applied because this pass emits
mattes, not RGB (comp does its own despill).

Same refiner contract as RVM/BiRefNet/ViTMatte (inherits the machinery:
all-masks mode, inter-object exclusion, compare-mode channel_suffix):
- outward: bounded by the dilated SAM mask (garbage matte);
- inward:  the eroded SAM core stays solid 1.0 — a subject wearing
  screen-coloured clothing (blue jeans on a blue screen) cannot punch a
  hole through its own matte.

Deterministic per-frame → no ML flicker → no smoothing declared.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from live_action_aov.core.pass_base import License
from live_action_aov.passes.matte.rvm import RVMRefinerPass

_log = logging.getLogger(__name__)


def _screen_likeness(frame_rgb: np.ndarray, region: np.ndarray, screen: str) -> float:
    """How screen-coloured a region is: mean colour-difference d in [0,1]."""
    if not region.any():
        return 0.0
    px = frame_rgb[region]
    if screen == "blue":
        d = px[:, 2] - np.maximum(px[:, 0], px[:, 1])
    else:
        d = px[:, 1] - np.maximum(px[:, 0], px[:, 2])
    return float(np.clip(d, 0.0, 1.0).mean())


class ChromaKeyPass(RVMRefinerPass):
    name = "chroma_key"
    version = "0.1.0"
    license = License(
        spdx="MIT",
        commercial_use=True,
        commercial_tool_resale=True,
        notes=(
            "Pure colour-difference maths (no model, no weights) — "
            "commercial-safe with no provenance caveats. Alpha quality on "
            "hair/motion blur is physically derived from the screen, which "
            "on a real green/blue screen typically beats ML matting."
        ),
    )

    # Deterministic per-frame maths — temporally stable by construction.
    smoothable_channels: list[str] = []

    DEFAULT_PARAMS: dict[str, Any] = {
        # "auto" samples the plate outside the SAM mask and picks the
        # dominant screen primary; "green"/"blue" force it.
        "screen": "auto",
        # alpha_key = 1 - clip(d * gain - lift, 0, 1). Higher gain = harder
        # pull (more opaque subject, less screen bleed); lift eats screen
        # noise near zero.
        "key_gain": 2.2,
        "key_lift": 0.02,
        # SAM-bound trimap: outward garbage bound + solid inward core.
        # The bound is generous by default — the SAM mask may come from a
        # coarse proxy, and the KEY (not the bound) draws the real edge.
        "bound_dilate": 20,
        "core_erode": 6,
    }

    # ------------------------------------------------------------------
    # No model to load — the "engine" is arithmetic.
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        self._model = object()  # sentinel: pass is always "loaded"

    # ------------------------------------------------------------------
    # Keying
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_screen(frame_rgb: np.ndarray, subject_mask: np.ndarray) -> str:
        """'green' or 'blue' from the mean colour OUTSIDE the subject."""
        bg = subject_mask < 0.5
        if not bg.any():  # degenerate: subject fills frame — sample all
            bg = np.ones_like(bg)
        mean = frame_rgb[bg].mean(axis=0)
        return "blue" if mean[2] > mean[1] else "green"

    def _key_alpha(self, frame_rgb: np.ndarray, screen: str) -> np.ndarray:
        """Colour-difference pull on one (H, W, 3) float frame -> alpha."""
        r, g, b = frame_rgb[..., 0], frame_rgb[..., 1], frame_rgb[..., 2]
        if screen == "blue":
            d = b - np.maximum(r, g)
        else:
            d = g - np.maximum(r, b)
        gain = float(self.params.get("key_gain", 2.2))
        lift = float(self.params.get("key_lift", 0.02))
        return 1.0 - np.clip(d * gain - lift, 0.0, 1.0)

    def _refine_instance(
        self,
        plate_stack: np.ndarray,
        hard_stack: np.ndarray,
    ) -> np.ndarray:
        """Input: plate (T, H, W, 3) float sRGB [0,1]; hard (T, H, W).
        Output: (T, H, W) float32 soft alpha (key bounded by SAM)."""
        import cv2

        self._load_model()
        T, H, W, _ = plate_stack.shape
        dil_px = max(int(self.params.get("bound_dilate", 20)), 0)
        ero_px = max(int(self.params.get("core_erode", 6)), 0)
        k_dil = np.ones((2 * dil_px + 1, 2 * dil_px + 1), np.uint8) if dil_px else None
        k_ero = np.ones((2 * ero_px + 1, 2 * ero_px + 1), np.uint8) if ero_px else None
        screen_param = str(self.params.get("screen", "auto")).lower()

        out = np.zeros((T, H, W), dtype=np.float32)
        screen = screen_param if screen_param in ("green", "blue") else ""
        for t in range(T):
            binm = (hard_stack[t] > 0.5).astype(np.uint8)
            if int(binm.sum()) == 0:
                continue
            if not screen:  # auto: detect once, on the first populated frame
                screen = self._detect_screen(plate_stack[t], binm)
                # Seed-direction guard: the #1 usage mistake is seeding the
                # SCREEN instead of the subject. If the masked interior is
                # decidedly more screen-coloured than the outside, say so —
                # the output will be garbage and the user should know why.
                inside = _screen_likeness(plate_stack[t], binm > 0, screen)
                alt = "blue" if screen == "green" else "green"
                inside_alt = _screen_likeness(plate_stack[t], binm > 0, alt)
                worst = max(inside, inside_alt)
                if worst > 0.25:
                    _log.warning(
                        "chroma_key: the SAM mask interior looks like a %s "
                        "screen (likeness %.2f) — you probably seeded the "
                        "SCREEN. Seed the SUBJECT instead; the key removes "
                        "the screen around it automatically.",
                        alt if inside_alt > inside else screen,
                        worst,
                    )
            alpha = self._key_alpha(plate_stack[t], screen).astype(np.float32)
            dil = cv2.dilate(binm, k_dil) if k_dil is not None else binm
            core = cv2.erode(binm, k_ero) if k_ero is not None else binm
            # Trimap combine: key draws the edge inside the SAM garbage
            # bound; the SAM core stays solid (screen-coloured clothing
            # can't hole the subject).
            out[t] = np.clip(
                np.maximum(alpha * dil.astype(np.float32), core.astype(np.float32)),
                0.0,
                1.0,
            )
        return out


__all__ = ["ChromaKeyPass"]
