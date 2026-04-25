# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: MIT

"""Channel naming contract (design §5.1).

The comper-visible, load-bearing truth for sidecar EXR channel names. Passes
return `{channel_name: array}` from `postprocess()`; the writer packs those
into the EXR using exactly these names, in this order. Changing anything
here is a pipeline-wide breaking change.

Channel sets are exposed as frozen tuples so they can be used in sets /
hashed / iterated without accidental mutation.
"""

from __future__ import annotations

# --- Depth ---
CH_Z = "Z"
CH_Z_RAW = "Z_raw"
CH_DEPTH_CONFIDENCE = "depth.confidence"
DEPTH_CHANNELS = (CH_Z, CH_Z_RAW, CH_DEPTH_CONFIDENCE)

# --- Normals (camera-space, [-1,1], unit-length per pixel) ---
CH_N_X = "N.x"
CH_N_Y = "N.y"
CH_N_Z = "N.z"
CH_NORMALS_CONFIDENCE = "normals.confidence"
NORMAL_CHANNELS = (CH_N_X, CH_N_Y, CH_N_Z, CH_NORMALS_CONFIDENCE)

# --- Position (camera-space, metres when depth is metric, else relative) ---
# Derived from Z + intrinsics via the `PositionFromDepth` post-processor —
# runs automatically whenever a depth pass is in the job. Pinhole math:
#   P.x = (u - cx) / fx * Z
#   P.y = (v - cy) / fy * Z
#   P.z = Z
# Consumed by relight gizmos, envRelight, 3DGS export, etc.
CH_P_X = "P.x"
CH_P_Y = "P.y"
CH_P_Z = "P.z"
POSITION_CHANNELS = (CH_P_X, CH_P_Y, CH_P_Z)

# --- Flow (pixels at plate resolution) ---
# The spec (§5.1) locks motion.x/y + back.x/y as the canonical names. Nuke's
# VectorBlur and the wider VFX ecosystem expect the "forward/backward .u/.v"
# convention, so we also emit those as aliases of the same data — Nuke then
# auto-wires its motion nodes without the comp artist picking channels
# manually. Two naming conventions, identical pixel values per channel.
CH_MOTION_X = "motion.x"
CH_MOTION_Y = "motion.y"
CH_BACK_X = "back.x"
CH_BACK_Y = "back.y"
CH_FLOW_CONFIDENCE = "flow.confidence"
# Explicit occlusion is `1 - flow.confidence`. `flow.confidence` alone
# works as a mask but compers kept asking "which one is occlusion?" —
# this gives them the obvious answer without an invert step.
CH_FLOW_OCCLUSION = "flow.occlusion"
CH_FORWARD_U = "forward.u"
CH_FORWARD_V = "forward.v"
CH_BACKWARD_U = "backward.u"
CH_BACKWARD_V = "backward.v"
FLOW_CHANNELS = (
    CH_MOTION_X,
    CH_MOTION_Y,
    CH_BACK_X,
    CH_BACK_Y,
    CH_FLOW_CONFIDENCE,
    CH_FLOW_OCCLUSION,
    CH_FORWARD_U,
    CH_FORWARD_V,
    CH_BACKWARD_U,
    CH_BACKWARD_V,
)

# --- Hero mattes (top-4 soft alpha) ---
CH_MATTE_R = "matte.r"
CH_MATTE_G = "matte.g"
CH_MATTE_B = "matte.b"
CH_MATTE_A = "matte.a"
MATTE_CHANNELS = (CH_MATTE_R, CH_MATTE_G, CH_MATTE_B, CH_MATTE_A)

# --- Ambient occlusion (derived, SSAO post-processor) ---
# Named under the `ao.` layer so Nuke groups it with its future
# siblings (ao.bent_normal when we add it, etc.) instead of landing
# in the "other" catch-all bucket next to un-layered channels.
CH_AO = "ao.a"
AO_CHANNELS = (CH_AO,)


# --- Semantic masks (dynamic — one per detected concept) ---
MASK_PREFIX = "mask."
"""Dynamic channels follow the `mask.<concept>` convention, e.g.
`mask.person`, `mask.vehicle`. They're declared at runtime by the matte
pass and written as-is by the ExrSidecarWriter."""


def is_mask_channel(name: str) -> bool:
    """True if `name` follows the `mask.<concept>` semantic-mask convention."""
    return name.startswith(MASK_PREFIX) and len(name) > len(MASK_PREFIX)


#: EXR layer/channel ordering that Nuke and most EXR viewers expect (design
#: §11.3, trap 8). Writers sort channels against this order before
#: emitting; unknown channels follow in insertion order.
CANONICAL_CHANNEL_ORDER: tuple[str, ...] = (
    *DEPTH_CHANNELS,
    *POSITION_CHANNELS,
    *NORMAL_CHANNELS,
    *AO_CHANNELS,
    *FLOW_CHANNELS,
    *MATTE_CHANNELS,
)


__all__ = [
    "AO_CHANNELS",
    "CANONICAL_CHANNEL_ORDER",
    "CH_AO",
    "CH_BACKWARD_U",
    "CH_BACKWARD_V",
    "CH_BACK_X",
    "CH_BACK_Y",
    "CH_DEPTH_CONFIDENCE",
    "CH_FLOW_CONFIDENCE",
    "CH_FLOW_OCCLUSION",
    "CH_FORWARD_U",
    "CH_FORWARD_V",
    "CH_MATTE_A",
    "CH_MATTE_B",
    "CH_MATTE_G",
    "CH_MATTE_R",
    "CH_MOTION_X",
    "CH_MOTION_Y",
    "CH_NORMALS_CONFIDENCE",
    "CH_N_X",
    "CH_N_Y",
    "CH_N_Z",
    "CH_P_X",
    "CH_P_Y",
    "CH_P_Z",
    "CH_Z",
    "CH_Z_RAW",
    "DEPTH_CHANNELS",
    "FLOW_CHANNELS",
    "MASK_PREFIX",
    "MATTE_CHANNELS",
    "NORMAL_CHANNELS",
    "POSITION_CHANNELS",
    "is_mask_channel",
]
