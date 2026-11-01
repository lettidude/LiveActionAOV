"""MoGe (Microsoft) - monocular metric geometry, vendored for the `moge3` pass.

Source: https://github.com/microsoft/MoGe @ 74fbce054ebed49800de42d0ad0e83495065719a
License: MIT (see LICENSE); `model/modules/dinov2` is Meta's DINOv2, Apache-2.0.

Only the inference path of MoGe-2 / MoGe-3 is kept (model/v2.py, v3.py,
utils.py, modules/, utils/geometry_*.py, utils/tools.py), verbatim. Vendored
instead of installed because the upstream package pins numpy>=2 and pulls
gradio/trimesh for its demo app; the model itself needs only torch, scipy,
utils3d-moge and FlexGEMM (MoGe-3's sparse refiner, pure Triton).
"""
