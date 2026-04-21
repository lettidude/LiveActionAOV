# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: MIT

"""Sidecar writers.

`SidecarWriter` is the ABC. v1 ships `ExrSidecarWriter` and a minimal
`JsonSidecarWriter`. v2a adds Alembic + Nuke script writers; v2b adds FBX
ASCII; v3 adds PLY for Gaussian Splats — all as siblings discovered via the
`live_action_aov.io.writers` entry-point group.
"""
