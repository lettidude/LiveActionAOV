# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: MIT

"""Preview what a run generated, without opening Nuke.

Right-click a shot -> "Preview output": pick a version, a layer
(depth, normals, each mask, the splat ...) and scrub frames. Mattes can
be shown over the plate. Camera / splat solves are summarised from their
JSON reports. Reads only the channels of the shown layer, downsampled to
the window, so scrubbing a 4K multi-layer EXR stays usable.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QDesktopServices, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from live_action_aov.core.output_layout import CAMERA_DIR, PASSES_DIR, SPLAT_DIR
from live_action_aov.gui.preview_loader import (
    _linear_to_srgb,
    _preview_to_linear,
    _proxy_resize,
    _qimage_from_rgb_float32,
    _resolve_frame_path,
)
from live_action_aov.gui.shot_state import ShotState

_FRAME_RE = re.compile(r"\.(\d+)\.(?=(?:crypto\.)?exr$)")
_DISPLAY_LONG_EDGE = 1600


@dataclass
class _Sequence:
    template: str  # path with `{frame}` in place of the frame number
    pad: int
    frames: list[int] = field(default_factory=list)
    channels: list[str] = field(default_factory=list)

    def path(self, frame: int) -> Path:
        return Path(self.template.replace("{frame}", f"{frame:0{self.pad}d}"))


@dataclass
class Layer:
    name: str
    seq: _Sequence
    channels: list[str]


def scan_run(run_dir: Path) -> list[Layer]:
    """Every displayable layer of a run (Cryptomatte excluded)."""
    seqs: dict[str, _Sequence] = {}
    for p in sorted((run_dir / PASSES_DIR).rglob("*.exr")):
        m = _FRAME_RE.search(p.name)
        if m is None or p.name.endswith(".crypto.exr"):
            continue
        tmpl = str(p.parent / (p.name[: m.start(1)] + "{frame}" + p.name[m.end(1) :]))
        seq = seqs.setdefault(tmpl, _Sequence(tmpl, len(m.group(1))))
        seq.frames.append(int(m.group(1)))
    layers: list[Layer] = []
    for seq in seqs.values():
        seq.frames.sort()
        seq.channels = _channel_names(seq.path(seq.frames[0]))
        groups: dict[str, list[str]] = {}
        for ch in seq.channels:
            if ch.lower().startswith("crypto"):
                continue
            layer = ch.rsplit(".", 1)[0] if "." in ch else "rgba"
            groups.setdefault(layer, []).append(ch)
        for n, chs in groups.items():
            if n.lower().startswith("mask"):  # one entry per object, not a 2-ch "image"
                layers += [Layer(c, seq, [c]) for c in chs]
            else:
                layers.append(Layer(n, seq, chs))
    return layers


def _channel_names(path: Path) -> list[str]:
    import OpenImageIO as oiio

    inp = oiio.ImageInput.open(str(path))
    if inp is None:
        return []
    try:
        return list(inp.spec().channelnames)
    finally:
        inp.close()


def read_layer(layer: Layer, frame: int) -> np.ndarray:
    """(H, W, C) float32 of just this layer's channels."""
    import OpenImageIO as oiio

    idx = [layer.seq.channels.index(c) for c in layer.channels]
    lo, hi = min(idx), max(idx) + 1
    inp = oiio.ImageInput.open(str(layer.seq.path(frame)))
    if inp is None:
        raise OSError(f"cannot open {layer.seq.path(frame)}")
    try:
        spec = inp.spec()
        px = inp.read_image(0, 0, lo, hi, oiio.FLOAT)
    finally:
        inp.close()
    arr = np.asarray(px, dtype=np.float32).reshape(spec.height, spec.width, hi - lo)
    return arr[..., [i - lo for i in idx]]


def layer_to_display(layer: Layer, px: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Display RGB in [0, 1], plus a single-channel matte (for the
    over-plate view) when the layer is one."""
    px = np.nan_to_num(px, nan=0.0, posinf=0.0, neginf=0.0)
    names = [c.rsplit(".", 1)[-1].lower() for c in layer.channels]
    lname = layer.name.lower()
    if px.shape[-1] == 1 or names == ["z"]:
        v = px[..., 0]
        if lname.startswith(("mask", "matte")) or (v.min() >= 0.0 and v.max() <= 1.0):
            m = np.clip(v, 0.0, 1.0)
            return np.repeat(m[..., None], 3, axis=-1), m
        lo, hi = np.percentile(v, [1.0, 99.0])
        g = np.clip((v - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        return np.repeat(g[..., None], 3, axis=-1), None
    if lname.startswith("n") and px.shape[-1] >= 3 and px[..., :3].min() < -0.05:
        return np.clip(px[..., :3] * 0.5 + 0.5, 0.0, 1.0), None
    if px.shape[-1] == 2:  # motion vectors: u, v -> red / green around grey
        s = max(float(np.percentile(np.abs(px), 99.0)), 1e-6)
        rg = np.clip(px / (2 * s) + 0.5, 0.0, 1.0)
        return np.concatenate([rg, np.full_like(rg[..., :1], 0.5)], axis=-1), None
    rgb = px[..., :3] if px.shape[-1] >= 3 else np.repeat(px[..., :1], 3, axis=-1)
    alpha = px[..., 3] if px.shape[-1] >= 4 and names[-1] == "alpha" else None
    disp = np.clip(_linear_to_srgb(np.clip(rgb, 0.0, None)), 0.0, 1.0)
    if alpha is not None and rgb.max() <= 1e-6:  # alpha-only rgba (a matte)
        a = np.clip(alpha, 0.0, 1.0)
        return np.repeat(a[..., None], 3, axis=-1), a
    return disp, None


def summary_lines(run_dir: Path) -> list[str]:
    """Camera + splat solve stats from their JSON reports."""
    out: list[str] = []
    for p in sorted((run_dir / CAMERA_DIR).glob("*.camera.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            q = d.get("quality") or {}
            n_ok = len(d.get("extrinsics") or [])
            n_bad = len(d.get("unregistered_frames") or [])
            err = q.get("aggregate_reprojection_error_px")
            fx = (d.get("intrinsics") or {}).get("fx_px")
            line = f"Camera: {n_ok} frames solved"
            line += f", {n_bad} lost" if n_bad else ""
            line += f", reprojection {float(err):.2f} px" if err is not None else ""
            line += f", focal {float(fx):.0f} px" if fx else ""
            out.append(line)
        except Exception:
            out.append(f"Camera: {p.name} (unreadable report)")
    for p in sorted((run_dir / SPLAT_DIR).glob("*.splat.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            bits = [f"{int(d['gaussians']):,} gaussians"] if d.get("gaussians") else []
            if d.get("holdout_psnr_db") is not None:
                bits.append(f"held-out PSNR {d['holdout_psnr_db']} dB")
            if (d.get("masked_fraction") or 0) > 0.4:
                bits.append(f"{100 * d['masked_fraction']:.0f}% masked: clean plate unreliable")
            if d.get("train_seconds") is not None:
                bits.append(f"{d['train_seconds']:.0f} s training")
            out.append("Splat: " + (", ".join(bits) or p.name))
        except Exception:
            out.append(f"Splat: {p.name} (unreadable report)")
    return out


class OutputPreviewDialog(QDialog):
    def __init__(
        self, shot: ShotState, versions: list[Path], parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Output preview - {shot.name}")
        self.resize(1280, 820)
        self._shot = shot
        self._layers: list[Layer] = []
        self._run: Path | None = None
        self._plate_cache: dict[int, np.ndarray] = {}

        self._version = QComboBox()
        for v in versions:
            self._version.addItem(v.name, v)
        self._version.currentIndexChanged.connect(self._load_version)
        self._layer_list = QListWidget()
        self._layer_list.currentRowChanged.connect(lambda _r: self._refresh())
        self._over_plate = QCheckBox("Matte over plate")
        self._over_plate.setChecked(True)
        self._over_plate.toggled.connect(lambda _c: self._refresh())
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._spin = QSpinBox()
        self._slider.valueChanged.connect(self._spin.setValue)
        self._spin.valueChanged.connect(self._slider.setValue)
        self._slider.valueChanged.connect(lambda _v: self._refresh())
        self._image = QLabel()
        self._image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image.setMinimumSize(640, 360)
        self._image.setStyleSheet("background: #111;")
        self._info = QLabel("")
        self._info.setWordWrap(True)
        self._info.setStyleSheet("color: #aaa;")
        open_btn = QPushButton("Open folder")
        open_btn.clicked.connect(lambda: self._open(self._run))
        nk_btn = QPushButton("Open in Nuke")
        nk_btn.clicked.connect(self._open_comp)

        side = QVBoxLayout()
        side.addWidget(QLabel("Version"))
        side.addWidget(self._version)
        side.addWidget(QLabel("Layer"))
        side.addWidget(self._layer_list, stretch=1)
        side.addWidget(self._over_plate)
        side.addWidget(open_btn)
        side.addWidget(nk_btn)
        frame_row = QHBoxLayout()
        frame_row.addWidget(self._slider, stretch=1)
        frame_row.addWidget(self._spin)
        main = QVBoxLayout()
        main.addWidget(self._image, stretch=1)
        main.addLayout(frame_row)
        main.addWidget(self._info)
        root = QHBoxLayout(self)
        side_w = QWidget()
        side_w.setLayout(side)
        side_w.setFixedWidth(240)
        root.addWidget(side_w)
        root.addLayout(main, stretch=1)
        self._load_version()

    def _load_version(self) -> None:
        self._run = self._version.currentData()
        self._layers = scan_run(self._run) if self._run else []
        self._layer_list.blockSignals(True)
        self._layer_list.clear()
        for layer in self._layers:
            self._layer_list.addItem(layer.name)
        self._layer_list.blockSignals(False)
        frames = sorted({f for layer in self._layers for f in layer.seq.frames})
        lo, hi = (frames[0], frames[-1]) if frames else (0, 0)
        for w in (self._slider, self._spin):
            w.blockSignals(True)
            w.setRange(lo, hi)
            w.setValue(lo)
            w.blockSignals(False)
        lines = summary_lines(self._run) if self._run else []
        if not self._layers:
            lines.insert(0, "No EXR passes in this version.")
        self._info.setText("\n".join(lines))
        if self._layers:
            self._layer_list.setCurrentRow(0)
        else:
            self._image.clear()

    def _plate(self, frame: int, shape: tuple[int, int]) -> np.ndarray | None:
        if frame not in self._plate_cache:
            try:
                from live_action_aov.io.oiio_io import read_exr

                px, _ = read_exr(_resolve_frame_path(self._shot, frame))
                lin = _preview_to_linear(px[..., :3], self._shot.effective_colorspace())
                self._plate_cache = {frame: np.clip(_linear_to_srgb(lin), 0.0, 1.0)}
            except Exception:
                return None
        p = self._plate_cache[frame]
        if p.shape[:2] != shape:
            import cv2

            p = cv2.resize(p, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
        return p

    def _refresh(self) -> None:
        row = self._layer_list.currentRow()
        if not (0 <= row < len(self._layers)):
            return
        layer = self._layers[row]
        frame = self._slider.value()
        if frame not in layer.seq.frames:
            self._image.setText(f"{layer.name}: no file for frame {frame}")
            return
        try:
            px = _proxy_resize(read_layer(layer, frame), _DISPLAY_LONG_EDGE)
        except Exception as e:
            self._image.setText(f"Could not read {layer.name}: {e}")
            return
        rgb, matte = layer_to_display(layer, px)
        if matte is not None and self._over_plate.isChecked():
            plate = self._plate(frame, (matte.shape[0], matte.shape[1]))
            if plate is not None:
                tint = np.array([1.0, 0.25, 0.2], dtype=np.float32)
                a = matte[..., None] * 0.6
                rgb = plate * (1 - a) + tint * a
        q = _qimage_from_rgb_float32(np.ascontiguousarray(rgb, dtype=np.float32))
        pm = QPixmap.fromImage(q).scaled(
            self._image.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._image.setPixmap(pm)

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        self._refresh()

    def _open_comp(self) -> None:
        if self._run is None:
            return
        comps = sorted(self._run.glob("*.comp.nk"))
        self._open(comps[0] if comps else self._run)

    @staticmethod
    def _open(path: Path | None) -> None:
        if path is None:
            return
        from PySide6.QtCore import QUrl

        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


__all__ = [
    "Layer",
    "OutputPreviewDialog",
    "layer_to_display",
    "read_layer",
    "scan_run",
    "summary_lines",
]
