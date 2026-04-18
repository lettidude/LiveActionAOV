"""VIDEO_CLIP pass helpers — window planning and overlap stitching.

DepthCrafter and NormalCrafter both operate on clips longer than their
native window size (~110 frames) via a sliding window with ~25-frame
overlap. This module owns the generic math so both passes stitch the same
way, and so the stitching can be unit-tested without spinning up a real
diffusers pipeline.
"""

from live_action_aov.shared.video_clip.sliding_window import (
    plan_window_starts,
    stitch_windowed_predictions,
    trapezoid_weight,
)

__all__ = ["plan_window_starts", "stitch_windowed_predictions", "trapezoid_weight"]
