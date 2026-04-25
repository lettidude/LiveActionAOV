# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: MIT

"""Central model registry — lazy loading + reference counting + VRAM tracking.

The registry is the single place where model checkpoints are loaded. Passes
call `ModelRegistry.get("depth_anything_v2_base")` instead of managing
their own loading, which lets the scheduler unload inactive models under
VRAM pressure and share weights between passes that use the same backbone
(design §14, decision 6).
"""
