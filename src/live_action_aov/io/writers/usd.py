# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: BUSL-1.1 (see LICENSE)
"""USD (`.usda`) writer for the camera track: the solved camera and a point
cloud, for Houdini, Maya, Blender, Unreal and Nuke's USD-based 3D system.

Plain text, written by hand: no USD library is needed to write it, and an
artist can open the file to see what is in it. One `Camera` prim animated
with a matrix per solved frame (frames the solve did not register have no
sample: the importer interpolates), and one `Points` prim with colours.

USD and the solve share conventions (Y up, right handed, the camera looks
down -Z), so nothing is flipped. USD matrices use row vectors: the camera's
axes are the rows and the translation is the last row.

Lens values are the ones the Nuke camera gets (focal and film back in mm).
USD reads them as tenths of a scene unit; an importer that converts scales
focal and film back together, so the field of view is the same either way.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from live_action_aov.io.camera_schema import CameraSolve
from live_action_aov.io.writers.display_cloud import DisplayCloud


def _focal_mm(solve: CameraSolve) -> float:
    intr = solve.intrinsics
    if intr.focal_mm_hint is not None:
        return float(intr.focal_mm_hint)
    return float(intr.fx_px * intr.haperture_mm_hint / max(intr.width_px, 1))


def _tuples(rows: np.ndarray, fmt: str) -> str:
    return ", ".join("(" + ", ".join(format(float(v), fmt) for v in row) + ")" for row in rows)


def render_usd(solve: CameraSolve, cloud: DisplayCloud | None = None, *, fps: float = 24.0) -> str:
    """The `.usda` text for `solve`, with `cloud` (else the solve's own points)."""
    frames = [e.frame for e in solve.extrinsics]
    first, last = (min(frames), max(frames)) if frames else (0, 0)
    q = solve.quality
    doc = (
        f"LiveActionAOV camera track ({solve.backend.name} {solve.backend.version}). "
        f"Units: {solve.conventions.units}. Reprojection error "
        f"{q.aggregate_reprojection_error_px:.3f} px over "
        f"{q.registered_frame_count}/{q.total_frame_count} frames."
    )
    if q.verdict:
        doc = f"{q.verdict}. {doc}"
    doc = doc.replace("\\", "/").replace('"', "'")
    name = re.sub(r"\W", "_", solve.shot.name or "shot")
    if not name or name[0].isdigit():
        name = f"shot_{name}"

    samples = []
    for e in solve.extrinsics:
        r = np.asarray(e.rotation_matrix, np.float64)
        t = e.translation
        rows = [
            (r[0][0], r[1][0], r[2][0], 0.0),
            (r[0][1], r[1][1], r[2][1], 0.0),
            (r[0][2], r[1][2], r[2][2], 0.0),
            (t[0], t[1], t[2], 1.0),
        ]
        body = ", ".join("(" + ", ".join(f"{v:.9g}" for v in row) + ")" for row in rows)
        samples.append(f"            {e.frame}: ( {body} ),")

    if cloud is not None and len(cloud):
        xyz, rgb, width = cloud.xyz, cloud.rgb, cloud.size
        what = cloud.note
    else:
        xyz = np.array([p.xyz for p in solve.points], np.float32).reshape(-1, 3)
        rgb = np.array([p.color_rgb for p in solve.points], np.float32).reshape(-1, 3) / 255.0
        span = float(np.ptp(xyz, axis=0).max()) if len(xyz) else 1.0
        width = np.full(len(xyz), span / 400.0, np.float32)
        what = f"{len(xyz)} points: feature points of the solve"

    lines = [
        "#usda 1.0",
        "(",
        f'    doc = "{doc}"',
        '    defaultPrim = "LiveActionAOV"',
        '    upAxis = "Y"',
        "    metersPerUnit = 1",
        f"    startTimeCode = {first}",
        f"    endTimeCode = {last}",
        f"    timeCodesPerSecond = {fps:g}",
        f"    framesPerSecond = {fps:g}",
        ")",
        "",
        'def Xform "LiveActionAOV" (',
        '    kind = "assembly"',
        ")",
        "{",
        f'    def Camera "{name}_camera"',
        "    {",
        '        token projection = "perspective"',
        f"        float focalLength = {_focal_mm(solve):.6f}",
        f"        float horizontalAperture = {solve.intrinsics.haperture_mm_hint:.6f}",
        f"        float verticalAperture = {solve.intrinsics.vaperture_mm_hint:.6f}",
        "        float2 clippingRange = (0.1, 10000)",
        "        matrix4d xformOp:transform.timeSamples = {",
        *samples,
        "        }",
        '        uniform token[] xformOpOrder = ["xformOp:transform"]',
        "    }",
    ]
    if len(xyz):
        lines += [
            "",
            f'    def Points "{name}_points" (',
            f'        doc = "{what.replace(chr(34), chr(39))}"',
            "    )",
            "    {",
            f"        point3f[] points = [{_tuples(xyz, '.5f')}]",
            f"        float[] widths = [{', '.join(f'{v:.5f}' for v in width)}] (",
            '            interpolation = "vertex"',
            "        )",
            f"        color3f[] primvars:displayColor = [{_tuples(rgb, '.4f')}] (",
            '            interpolation = "vertex"',
            "        )",
            "    }",
        ]
    lines += ["}", ""]
    return "\n".join(lines)


def write_usd(
    path: Path, solve: CameraSolve, cloud: DisplayCloud | None = None, *, fps: float = 24.0
) -> Path:
    path.write_text(render_usd(solve, cloud, fps=fps), encoding="utf-8", newline="\n")
    return path


__all__ = ["render_usd", "write_usd"]
