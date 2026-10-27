"""QC a camera_pycolmap solve INSIDE Nuke, with Nuke's own camera maths.

Two steps (the first runs in the tool's venv, the second in Nuke):

    .venv\\Scripts\\python.exe scripts\\nuke_check_camera.py dump <camera_track_dir> <shot>
    "C:\\Program Files\\Nuke16.0v8\\Nuke16.0.exe" -t scripts\\nuke_check_camera.py check <camera_track_dir> <shot>

`dump` samples observations from the COLMAP model the pass wrote
(`<shot>_colmap/`) - each one a solved 3D point (in the exported Y-up world)
plus the pixel where COLMAP saw it. `check` pastes `<shot>.camera.nk`,
evaluates the Camera2's world_matrix at each frame (Nuke's own ZXY Euler
-> matrix), projects the points with focal/haperture, and reports the
pixel error. It should match the solve's reprojection error; a big gap
means the export (axes, Euler order, film back) is wrong.

Validated on ACTIONVFX PF0081 (2048x1080 dolly): 0.99 px mean in Nuke 16,
same as the solve.
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path


def dump(track_dir: Path, shot: str, n: int = 400) -> None:
    import pycolmap

    solve = json.loads((track_dir / f"{shot}.camera.json").read_text(encoding="utf-8"))
    rec = pycolmap.Reconstruction(str(track_dir / f"{shot}_colmap"))
    cam = next(iter(rec.cameras.values()))
    s = solve["intrinsics"]["width_px"] / cam.width
    obs = []
    for im in rec.images.values():
        frame = int(Path(im.name).stem.split("_")[-1])
        for p in im.points2D:
            if p.has_point3D():
                X = rec.points3D[p.point3D_id].xyz
                # Same 180 deg X world flip the exporter applies.
                obs.append(
                    {
                        "frame": frame,
                        "X": [float(X[0]), -float(X[1]), -float(X[2])],
                        "xy": [float(p.xy[0] * s), float(p.xy[1] * s)],
                    }
                )
    random.seed(0)
    obs = random.sample(obs, min(n, len(obs)))
    meta = {"width": solve["intrinsics"]["width_px"], "height": solve["intrinsics"]["height_px"]}
    (track_dir / "nuke_check_obs.json").write_text(json.dumps({"meta": meta, "obs": obs}), encoding="utf-8")
    print(f"{len(obs)} observations -> {track_dir / 'nuke_check_obs.json'}")


def check(track_dir: Path, shot: str) -> None:
    import nuke  # type: ignore[import-not-found]

    data = json.loads((track_dir / "nuke_check_obs.json").read_text(encoding="utf-8"))
    W, H = float(data["meta"]["width"]), float(data["meta"]["height"])
    nuke.nodePaste(str(track_dir / f"{shot}.camera.nk").replace("\\", "/"))
    cam = next(n for n in nuke.allNodes() if n.Class().startswith("Camera"))
    fpx = cam["focal"].getValue() / cam["haperture"].getValue() * W
    errs = []
    for o in data["obs"]:
        # world_matrix evaluated AT the frame (nukescripts.snap3d reads the
        # "current" frame, which nuke.frame() does not move in -t mode).
        wm = cam["world_matrix"].getValueAt(o["frame"])
        M = nuke.math.Matrix4()
        for i in range(16):
            M[i] = wm[i]
        M.transpose()
        c = M.inverse().transform(nuke.math.Vector3(*o["X"]))
        x = W / 2 + fpx * c.x / -c.z
        y = H / 2 - fpx * c.y / -c.z  # image y down
        errs.append(math.hypot(x - o["xy"][0], y - o["xy"][1]))
    errs.sort()
    print(
        f"{cam.name()}: N {len(errs)}  mean {sum(errs) / len(errs):.3f} px  "
        f"median {errs[len(errs) // 2]:.3f} px  p90 {errs[int(len(errs) * 0.9)]:.3f} px"
    )


if __name__ == "__main__":
    mode, d, shot_name = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    (dump if mode == "dump" else check)(d, shot_name)
