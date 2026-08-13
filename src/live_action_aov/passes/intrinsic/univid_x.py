# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: MIT

"""UniVidX intrinsic pass — albedo + irradiance (commercial-safe, video-native).

Status: **contract declared, backend NOT yet wired.**
-----------------------------------------------------

This file declares the full pass contract (channels, license, temporal
mode, VRAM estimate) so the integration shape is fixed and testable, but
`_load_model` deliberately raises `NotImplementedError`. The model
backend (a vendored Wan2.1-based diffusion pipeline) is intentionally
NOT vendored yet — the 14B UniVidX model has to be validated on real
hardware first (quality + runtime + VRAM) before we commit ~40 vendored
files to the tree. See:

  - `scripts/poc_unividx_prep.py` — prep harness: turns a plate sequence
    into the display-space frames UniVidX consumes, so you can run the
    upstream inference on your own GPU and eyeball the albedo.
  - `docs/albedo-unividx.md` — the validation recipe + vendoring plan.

This mirrors the repo's existing precedent (the MatAnyone2 refiner ships
its contract but raises `NotImplementedError` in `_refine_instance`
until the real backend lands, and is withheld from the GUI catalog so
users never see a broken option).

Why UniVidX for albedo
----------------------

Intrinsic decomposition (albedo × shading) is the holy grail for relight:
swap the shading, keep the base colour. As of 2026-05 UniVidX is the only
*commercial-safe* video-native intrinsic model — Apache-2.0 code AND
Apache-2.0 weights, built on Apache-2.0 Wan2.1-T2V. The alternatives are
all blocked: V-RGBX (CC-BY-NC weights), UniRelight (NVIDIA noncommercial,
and it needs a target HDRI — it's a relighter, not an extractor), and the
Careaga/Aksoy Ordinal/Colorful intrinsic models (academic-only license).

Outputs (spec §5.1, channels.py)
--------------------------------
- `albedo.r/g/b`     — view-independent base colour, lighting removed.
- `irradiance.r/g/b` — incident-light / shading term. Lambertian
  approximation: plate ≈ albedo * irradiance.

Both emitted in a linear working space so a comper can divide the plate
by albedo to recover shading, or relight by replacing irradiance.

Temporal: VIDEO_CLIP. UniVidX is a video diffusion model — it denoises
the whole clip jointly, which is the entire point (temporally consistent
albedo, no per-frame flicker). No smoother wiring needed.
"""

from __future__ import annotations

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
    CH_IRRADIANCE_B,
    CH_IRRADIANCE_G,
    CH_IRRADIANCE_R,
)

# VRAM floor for the 14B Wan2.1 backbone UniVidX rides on. An early Q8
# field report suggested ~15 GB, but the UniVidX_ComfyUI maintainer (same
# upstream model) reports a hard 24 GB minimum at FP8 — peak 18-20 GB FP8,
# 32-34 GB BF16. So 24 is the honest floor; a 16 GB claim would let the
# GUI offer this pass on a card that then OOM-crashes mid-run. Used by the
# GUI VRAM capability gate (gui/cuda_check.meets_vram_requirement) so this
# pass bows out gracefully on small cards instead of crashing.
MIN_VRAM_GB = 24.0


class UniVidXIntrinsicPass(UtilityPass):
    name = "univid_x_intrinsic"
    version = "0.0.1"  # contract-only; bumps to 0.1.0 when backend lands
    license = License(
        spdx="Apache-2.0",
        commercial_use=True,
        commercial_tool_resale=True,
        notes=(
            "UniVidX is Apache-2.0 (HKUST MMLab et al., 2026) — code AND "
            "weights — built on Apache-2.0 Wan2.1-T2V-14B (Alibaba). The "
            "full chain is commercial-safe, unlike V-RGBX (NC weights), "
            "UniRelight (NVIDIA NC), and the Careaga/Aksoy intrinsic models "
            "(academic-only)."
        ),
    )
    pass_type = PassType.RADIOMETRIC
    temporal_mode = TemporalMode.VIDEO_CLIP
    # UniVidX is trained at 480p clips; surfaced so the scheduler can size
    # its frame buffer. Refine once the backend is validated.
    temporal_window = 57
    input_colorspace = "srgb_display"

    # Heavy model → declare a VRAM floor so the capability gate can hide
    # this pass on cards that can't run it.
    @staticmethod
    def vram_estimate_gb_fn(w: int, h: int) -> float:
        del w, h  # constant ceiling — the 14B backbone dominates, not plate size
        return MIN_VRAM_GB

    produces_channels = [
        ChannelSpec(name=CH_ALBEDO_R, description="Albedo (base colour) R, lighting removed"),
        ChannelSpec(name=CH_ALBEDO_G, description="Albedo (base colour) G, lighting removed"),
        ChannelSpec(name=CH_ALBEDO_B, description="Albedo (base colour) B, lighting removed"),
        ChannelSpec(name=CH_IRRADIANCE_R, description="Irradiance (shading) R"),
        ChannelSpec(name=CH_IRRADIANCE_G, description="Irradiance (shading) G"),
        ChannelSpec(name=CH_IRRADIANCE_B, description="Irradiance (shading) B"),
    ]
    # VIDEO_CLIP → already temporally coherent; nothing for the smoother.
    smoothable_channels: list[str] = []

    DEFAULT_PARAMS: dict[str, Any] = {
        "precision": "fp8",  # "fp8" | "fp16" — fp8 keeps it inside 24 GB
        "input_size": 480,  # UniVidX training resolution (480p)
        "seed": 0,
    }

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        for k, v in self.DEFAULT_PARAMS.items():
            self.params.setdefault(k, v)
        self._model: Any = None
        self._device: Any = None

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load the vendored UniVidX pipeline.

        NOT YET IMPLEMENTED. The backend is vendored only after the POC
        validates quality + runtime on real hardware — see this module's
        docstring, `scripts/poc_unividx_prep.py`, and
        `docs/albedo-unividx.md`. Raising here (rather than shipping a
        guessed-at pipeline) keeps the tree honest: the contract is real,
        the inference is not wired.

        Phase 3 contract — weights are LAZY. The ~85 GB of weights
        (Wan2.1-T2V-14B backbone + UniVidX LoRAs) must download on the
        FIRST run of this pass, never at install time, and only after the
        user has been warned about the size. Installing LiveActionAOV must
        stay lightweight; a multi-tens-of-GB pull at install would be a
        non-starter for the field. See docs/albedo-unividx.md §"Weights are
        lazy" for the exact requirement.
        """
        raise NotImplementedError(
            "UniVidXIntrinsicPass backend is not wired yet. Validate the "
            "model with scripts/poc_unividx_prep.py on a 24 GB+ GPU first, "
            "then vendor the Apache-2.0 inference under "
            "src/live_action_aov/vendored/univid_x/ (see docs/albedo-unividx.md)."
        )

    # ------------------------------------------------------------------
    # Single-frame lifecycle (stubs — VIDEO_CLIP drives via run_shot).
    # ------------------------------------------------------------------

    def preprocess(self, frames: np.ndarray) -> Any:
        return frames

    def infer(self, tensor: Any) -> Any:
        raise NotImplementedError("UniVidXIntrinsicPass is VIDEO_CLIP; drive it via run_shot.")

    def postprocess(self, tensor: Any) -> dict[str, np.ndarray]:
        raise NotImplementedError("UniVidXIntrinsicPass is VIDEO_CLIP; drive it via run_shot.")

    # ------------------------------------------------------------------
    # Shot-level iteration
    # ------------------------------------------------------------------

    def run_shot(
        self,
        reader: Any,
        frame_range: tuple[int, int],
    ) -> dict[int, dict[str, np.ndarray]]:
        # Fails fast at _load_model until the backend is wired. Kept as a
        # real method (not abstract-stub) so the integration shape is
        # locked: read frames → _infer_clip → per-frame albedo/irradiance.
        self._load_model()
        raise NotImplementedError  # unreachable until _load_model is wired


__all__ = ["MIN_VRAM_GB", "UniVidXIntrinsicPass"]
