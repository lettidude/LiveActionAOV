# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: MIT

"""Image sequence readers.

`ImageSequenceReader` is the ABC; `OIIOExrReader` is the v1 concrete
implementation. DPX / MOV / R3D readers will be added as siblings in v2
without touching core (design §14, decision 3).
"""
