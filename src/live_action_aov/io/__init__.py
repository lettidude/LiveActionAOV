# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: MIT

"""I/O, colorspace, resize, display transform, metadata.

All pixel I/O goes through OpenImageIO; all colorspace conversion goes
through OpenColorIO. Other modules never touch raw EXR bytes or do manual
colorspace math.
"""
