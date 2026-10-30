# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: BUSL-1.1 (see LICENSE)

"""Comp starter script (`<shot>.comp.nk`): every output of a run, wired up.

Open it in Nuke and everything the tool wrote is already there:

- the plate, and one Read per utility EXR (one multi-layer Read, or one per
  pass folder with the per-pass layout), all relative to the script folder
  so the output folder can be moved or copied to another machine;
- one Shuffle2 per pass, labelled (each mask on its own, depth/normals/flow
  to RGB, mattes and the splat clean plate to RGBA), inside a backdrop per
  pass;
- the camera track (Camera2) with its point cloud (ReadGeo2) in a Scene and
  a ScanlineRender over the plate - the usual matchmove check;
- with a splat + masks: a Keymix that swaps the first masked object for the
  splat clean plate, as a ready example of the clean-up.

Nuke quirks this has to follow (each one verified with Nuke 16 `-t`):

- our `mask.*` layer collides with Nuke's built-in `mask` layer, so Nuke
  shows it as `mask_extra.*`; likewise `motion.x/y` -> `motion_extra`,
  `depth.confidence` -> `depth_extra`, a bare `Z_raw` -> `other.Z_raw`;
- channel suffixes r/g/b/a (and R/G/B/A) read as red/green/blue/alpha;
- custom layers must be declared with `add_layer` at the top, or a Shuffle2
  parsed before its Read is evaluated drops its mappings to black.

Nothing here needs Nuke installed: the script is plain text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# Nuke's built-in layers and their channels. A channel of ours in one of
# these layers that is NOT a built-in channel lands in `<layer>_extra`.
_BUILTIN: dict[str, tuple[str, ...]] = {
    "rgba": ("red", "green", "blue", "alpha"),
    "rgb": ("red", "green", "blue"),
    "alpha": ("alpha",),
    "depth": ("Z",),
    "motion": ("u", "v"),
    "forward": ("u", "v"),
    "backward": ("u", "v"),
    "mask": ("a",),
}
_SUFFIX = {"r": "red", "g": "green", "b": "blue", "a": "alpha"}
_RGBA = ("red", "green", "blue", "alpha")

#: Pass folder (per-pass layout) and backdrop group for each EXR layer.
_GROUP_OF_LAYER = {
    "Z": "depth",
    "Z_raw": "depth",
    "depth": "depth",
    "N": "normals",
    "normals": "normals",
    "P": "position",
    "motion": "flow",
    "back": "flow",
    "forward": "flow",
    "backward": "flow",
    "flow": "flow",
    "ao": "ao",
    "albedo": "intrinsic",
    "irradiance": "intrinsic",
    "splat": "splat",
    "key": "matte",
    "matte": "matte",
    "mask": "matte",
}

_GROUP_COLOR = {
    "matte": 0x7F4C4CFF,
    "depth": 0x4C5F7FFF,
    "normals": 0x5F4C7FFF,
    "position": 0x5F4C7FFF,
    "flow": 0x4C7F6AFF,
    "splat": 0x7F6A4CFF,
    "intrinsic": 0x7F7F4CFF,
    "cryptomatte": 0x6A6A6AFF,
    "camera": 0x4C6A7FFF,
}


def pass_group(channel: str) -> str:
    """Which pass a sidecar channel belongs to (per-pass folder name)."""
    layer = channel.split(".", 1)[0]
    if layer.lower().startswith("crypto"):
        return "cryptomatte"
    if layer.startswith("mask") or layer.startswith("matte"):
        return "matte"
    return _GROUP_OF_LAYER.get(layer, layer if "." in channel else "other")


def nuke_channel(channel: str) -> str:
    """Our EXR channel name -> the name Nuke gives it on read."""
    layer, dot, chan = channel.partition(".")
    if not dot:
        return "depth.Z" if channel == "Z" else f"other.{channel}"
    chan = _SUFFIX.get(chan.lower(), chan) if len(chan) == 1 else chan
    builtin = _BUILTIN.get(layer)
    if builtin is not None and chan not in builtin:
        layer = f"{layer}_extra"
    return f"{layer}.{chan}"


@dataclass
class CompRead:
    """One utility Read: a path relative to the script and its channels."""

    group: str  # backdrop / folder label ("all passes", "depth", ...)
    rel_pattern: str  # e.g. "shot.utility.####.exr" or "depth/shot.depth.####.exr"
    channels: list[str] = field(default_factory=list)  # OUR channel names


def _q(s: str) -> str:
    return s.replace("\\", "/").replace('"', '\\"')


def _node_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", s)


class _Script:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.n = 0

    def node(self, cls: str, knobs: list[str], *, inputs: int | None = None) -> None:
        self.lines.append(f"{cls} {{")
        if inputs is not None:
            self.lines.append(f" inputs {inputs}")
        self.lines.extend(f" {k}" for k in knobs)
        self.lines.append("}")

    def set_var(self, var: str) -> None:
        self.lines.append(f"set {var} [stack 0]")

    def push(self, var: str | None) -> None:
        self.lines.append(f"push ${var}" if var else "push 0")


def _shuffles_for(channels: list[str]) -> list[tuple[str, str, list[tuple[str, str]]]]:
    """(label, in1 layer, [(src, dst)]) for every view worth a Shuffle2."""
    by_layer: dict[str, list[str]] = {}
    for ch in channels:
        n = nuke_channel(ch)
        by_layer.setdefault(n.split(".", 1)[0], []).append(n)
    out: list[tuple[str, str, list[tuple[str, str]]]] = []
    for layer, chans in by_layer.items():
        if layer.lower().startswith("crypto"):
            continue  # the Cryptomatte node reads those
        if layer.startswith("mask"):
            # One view per object: its matte in all four channels.
            for c in chans:
                out.append((c.split(".", 1)[1], layer, [(c, f"rgba.{d}") for d in _RGBA]))
            continue
        names = {c.split(".", 1)[1]: c for c in chans}
        rgba = [names[d] for d in _RGBA if d in names]
        rest = [c for c in chans if c not in rgba]
        if rgba:
            out.append((layer, layer, [(c, f"rgba.{c.split('.', 1)[1]}") for c in rgba]))
        if len(rest) > 1 and not rgba:
            out.append(
                (layer, layer, [(c, f"rgba.{d}") for c, d in zip(rest, _RGBA[:3], strict=False)])
            )
        else:
            for c in rest:
                out.append((c.replace(".", " "), layer, [(c, f"rgba.{d}") for d in _RGBA[:3]]))
    return out


def _add_layers(channels: list[str]) -> list[str]:
    """`add_layer` lines for every non-built-in layer we reference."""
    layers: dict[str, list[str]] = {}
    for ch in channels:
        n = nuke_channel(ch)
        layer = n.split(".", 1)[0]
        if layer in _BUILTIN:
            continue
        layers.setdefault(layer, [])
        if n not in layers[layer]:
            layers[layer].append(n)
    return [f"add_layer {{{layer} {' '.join(chs)}}}" for layer, chs in layers.items()]


def _mappings(pairs: list[tuple[str, str]], in_layer_order: list[str]) -> str:
    items = []
    for src, dst in pairs:
        src_idx = in_layer_order.index(src) if src in in_layer_order else 0
        dst_idx = _RGBA.index(dst.split(".", 1)[1])
        items.append(f"{src} 0 {src_idx} {dst} 0 {dst_idx}")
    # Unmapped rgba outputs go black (Shuffle2 always lists all four).
    mapped = {dst for _, dst in pairs}
    for d in _RGBA:
        if f"rgba.{d}" not in mapped:
            items.append(f"black -1 -1 rgba.{d} 0 {_RGBA.index(d)}")
    return f'"{len(items)} ' + " ".join(items) + '"'


def render_comp_script(
    *,
    shot_name: str,
    plate_pattern: str,
    frame_range: tuple[int, int],
    resolution: tuple[int, int],
    pixel_aspect: float,
    reads: list[CompRead],
    camera_nk: Path | None = None,
    points_obj_rel: str | None = None,
    crypto_rel_pattern: str | None = None,
) -> str:
    first, last = int(frame_range[0]), int(frame_range[1])
    w, h = int(resolution[0]), int(resolution[1])
    fmt = f"{w} {h} 0 0 {w} {h} {pixel_aspect:g} LiveActionAOV_{_node_name(shot_name)}"
    here = "[file dirname [value root.name]]"
    all_channels = [c for r in reads for c in r.channels]
    s = _Script()
    s.lines += [
        "#! nuke",
        "# Generated by LiveActionAOV: every output of this run, wired up.",
        "# Paths are relative to this script - keep it inside the output folder.",
        "Root {",
        " inputs 0",
        f" first_frame {first}",
        f" last_frame {last}",
        f' format "{fmt}"',
        "}",
        *_add_layers(all_channels),
    ]

    read_knobs = [
        f'format "{w} {h} 0 0 {w} {h} {pixel_aspect:g} "',
        f"first {first}",
        f"last {last}",
        f"origfirst {first}",
        f"origlast {last}",
        "origset true",
    ]
    # --- plate ---
    s.node(
        "Read",
        [
            f'file "{_q(plate_pattern)}"',
            *read_knobs,
            "name Plate",
            "label PLATE",
            "xpos 0",
            "ypos -400",
        ],
        inputs=0,
    )
    s.set_var("Nplate")

    x = 250
    mask_var: str | None = None
    splat_var: str | None = None
    backdrops: list[tuple[str, int, int, int, int]] = []
    for ri, rd in enumerate(reads):
        rname = f"Read_{_node_name(rd.group)}"
        s.node(
            "Read",
            [
                f'file "{here}/{_q(rd.rel_pattern)}"',
                *read_knobs,
                "raw true",  # data passes: never colour-convert
                f"name {rname}",
                f'label "{_q(rd.group)}"',
                f"xpos {x}",
                "ypos -400",
            ],
            inputs=0,
        )
        rvar = f"Nread{ri}"
        s.set_var(rvar)
        order: dict[str, list[str]] = {}
        for ch in rd.channels:
            n = nuke_channel(ch)
            order.setdefault(n.split(".", 1)[0], []).append(n)
        sx = x
        shuffles = _shuffles_for(rd.channels)
        for k, (label, layer, pairs) in enumerate(shuffles):
            if k:
                s.push(rvar)
            sname = _node_name(f"{layer}_{label}")
            s.node(
                "Shuffle2",
                [
                    "fromInput1 {{0} B}",
                    f"in1 {layer}",
                    "fromInput2 {{0} B}",
                    f"mappings {_mappings(pairs, order.get(layer, []))}",
                    f"name {sname}",
                    f'label "{_q(label)}"',
                    f"xpos {sx}",
                    "ypos -250",
                ],
            )
            if layer == "splat" and any(d == "rgba.alpha" for _, d in pairs) and splat_var is None:
                splat_var = "Nsplat"
                s.set_var(splat_var)
            sx += 110
        # All objects in one matte (max of every mask): handy on its own and
        # the mask of the clean-plate example below.
        masks = [n for n in (nuke_channel(c) for c in rd.channels) if n.startswith("mask")]
        if masks and mask_var is None:
            s.push(rvar)
            expr = masks[0]
            for m in masks[1:]:
                expr = f"max({expr}, {m})"
            s.node(
                "Expression",
                [
                    *(f'expr{i} "{expr}"' for i in range(4)),
                    "name All_objects_matte",
                    'label "every mask (max)"',
                    f"xpos {sx}",
                    "ypos -250",
                ],
            )
            mask_var = "Nmask"
            s.set_var(mask_var)
            sx += 110
        width = max(sx - x, 110)
        backdrops.append((rd.group, x - 20, -460, width + 20, 260))
        x += width + 60

    # --- cryptomatte (lossless sibling file, or inside the multi-layer EXR) ---
    if crypto_rel_pattern:
        s.node(
            "Read",
            [
                f'file "{here}/{_q(crypto_rel_pattern)}"',
                *read_knobs,
                "raw true",
                "name Read_cryptomatte",
                f"xpos {x}",
                "ypos -400",
            ],
            inputs=0,
        )
        s.node("Cryptomatte", ["name Cryptomatte_objects", f"xpos {x}", "ypos -250"])
        backdrops.append(("cryptomatte", x - 20, -460, 130, 260))
        x += 190

    # --- splat clean-up example: plate with the first masked object swapped
    # for the splat clean plate ---
    if splat_var and mask_var:
        s.push(mask_var)
        s.push(splat_var)
        s.push("Nplate")
        s.node(
            "Keymix",
            [
                "channels rgba",
                "name CleanPlate_example",
                'label "plate with every masked object\\nreplaced by the splat clean plate\\n(check splat.alpha: unseen areas are empty)"',
                "xpos 0",
                "ypos -50",
            ],
            inputs=3,
        )

    # --- camera track + point cloud + matchmove check ---
    if camera_nk is not None and camera_nk.exists():
        cam_text = camera_nk.read_text(encoding="utf-8")
        start = cam_text.find("Camera2 {")
        if start >= 0:
            cx = x
            cam = cam_text[start:].rstrip()
            cam = re.sub(r"\n xpos [^\n]*|\n ypos [^\n]*", "", cam)
            cam = cam[:-1].rstrip() + f"\n xpos {cx + 220}\n ypos -250\n}}"
            s.lines.append(cam)
            s.set_var("Ncam")
            scene_in = 0
            if points_obj_rel:
                s.node(
                    "ReadGeo2",
                    [
                        f'file "{here}/{_q(points_obj_rel)}"',
                        "name PointCloud",
                        'label "camera track points"',
                        f"xpos {cx}",
                        "ypos -400",
                    ],
                    inputs=0,
                )
                scene_in = 1
            s.node("Scene", ["name Scene_track", f"xpos {cx + 10}", "ypos -280"], inputs=scene_in)
            s.set_var("Nscene")
            s.push("Ncam")
            s.push("Nscene")
            s.push("Nplate")
            s.node(
                "ScanlineRender",
                [
                    "name Render_track_check",
                    'label "plate + tracked camera\\n(add Cards/geo to the Scene)"',
                    f"xpos {cx}",
                    "ypos -150",
                ],
                inputs=3,
            )
            backdrops.append(("camera", cx - 20, -460, 340, 380))

    bd_lines: list[str] = []
    for i, (label, bx, by, bw, bh) in enumerate(backdrops):
        bd_lines += [
            "BackdropNode {",
            " inputs 0",
            f" name Backdrop_{i}_{_node_name(label)}",
            f" tile_color 0x{_GROUP_COLOR.get(label.split('/')[0], 0x555555FF):08x}",
            f' label "<center><b>{_q(label)}</b>"',
            " note_font_size 24",
            f" xpos {bx}",
            f" ypos {by}",
            f" bdwidth {bw}",
            f" bdheight {bh}",
            "}",
        ]
    # Backdrops first so they sit behind the nodes in the DAG.
    head_end = s.lines.index("}") + 1 + len(_add_layers(all_channels))
    s.lines[head_end:head_end] = bd_lines
    return "\n".join(s.lines) + "\n"


def write_comp_script(path: Path, **kwargs: object) -> Path:
    path.write_text(render_comp_script(**kwargs), encoding="utf-8")  # type: ignore[arg-type]
    return path


__all__ = ["CompRead", "nuke_channel", "pass_group", "render_comp_script", "write_comp_script"]
