# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: BUSL-1.1 (see LICENSE)

"""LTX-2.5 Alpha-Gen refiner: SAM 3 finds the object, LTX generates its matte.

A generative video matter, not an edge refiner: it is given only RGB, never
another model's alpha (it needs context to understand what is foreground).

`mode="whole_frame"` (the default) runs LTX ONCE over the full frame (per
chunk, not per object) and gives each object the foreground nearest to it:
every foreground pixel goes to the closest SAM track (distance transforms,
with a `seam_px` feathered seam where two objects touch); foreground farther
than `far_frac` of the width from every track belongs to no object and is
only in the `foreground.a` layer (props, blood, untracked people). No SAM
gate cuts LTX's outline, so a shoulder SAM missed stays in, and N objects
cost one LTX run instead of N.

`crop_labels`: objects named here are NOT taken from the whole-frame run.
Each gets its own LTX run on a RECTANGLE around it (below): for the one
object that needs more pixels than the whole frame gives it (a small or far
subject, fine hair on the hero). The others keep the whole-frame matte.

`mode="per_object"` does that crop run for every object. Per track:

1. The shot is cut in chunks of `chunk` frames (8n+1, LTX's frame grid),
   overlapping by `overlap`; each chunk gets its own box, so the crop follows
   a moving subject. Box = the object's hard mask over the chunk, padded by
   `pad`, grown so neighbouring objects of similar size stay WHOLE (a person
   cut at the box edge comes back as a hard line + texture garbage). A
   neighbour that is SCENERY (sky, building, tree: `scenery` in the SAM
   artifact), or bigger than `neighbour_max_ratio` times the padded box,
   never grows the crop: with the default concepts every object's box used to
   swallow the buildings and trees around it until it was the whole frame,
   and LTX, which mattes ONE salient foreground per crop, gave each object
   somebody else's alpha (A65 couple-car plate, 2026-11-09). Scenery tracks
   themselves are not sent to LTX in all-masks mode: SAM's edge, lightly
   anti-aliased, like the small objects under `refine_min_px`.
2. The crop is scaled to `long_edge` (multiple of 32) and sent to the user's
   ComfyUI (`comfy_url`) - LTX-2.5 22B distilled + the alpha-gen IC-LoRA.
   Model files are found on the server: the int8 safetensors if present,
   else a GGUF of the same model (ComfyUI-GGUF), so a smaller GPU only needs
   a quantised download. A ComfyUI on this machine reads the crops from a
   temp folder; a remote one (the studio's AI box) gets them uploaded.
3. The result is levelled (0.5 / 99.5 percentile: Alpha-Gen leaves a haze),
   put back where it came from at plate scale, and cross-faded with the
   neighbouring chunks over the overlap.
4. LTX mattes ANY foreground (a car door, the person next to it), so the
   matte is gated by ViTMatte's guided trimap of the SAM mask: wide where
   BiRefNet is unsure (hair, motion blur), tight where it is sure of
   background. (`gate="dilate"`: the SAM mask grown by `gate_frac`, which
   left soft fog over whatever LTX matted next to the object.)

`cache_dir`: each LTX call's raw result is saved there, keyed by a hash of
the frames sent and the models / settings, so a rerun (a crash later in the
shot, a change downstream) never repeats the diffusion.

ComfyUI not running, or a node / model missing: the object falls back to
ViTMatte with a warning - the run never fails because of it.

Licence: LTX-2 Community License - free for organisations under USD 10M
annual revenue, a commercial licence from Lightricks above that. Stated in
the GUI label and in the sidecar metadata.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from live_action_aov.core.pass_base import License
from live_action_aov.core.work_dir import make_work_dir
from live_action_aov.io.channels import CH_MATTE_A, CH_MATTE_B, CH_MATTE_G, CH_MATTE_R
from live_action_aov.passes.matte.boxmask import stack_box
from live_action_aov.passes.matte.rank import is_scenery_label
from live_action_aov.passes.matte.rvm import RVMRefinerPass

if TYPE_CHECKING:
    from live_action_aov.passes.matte.vitmatte import ViTMatteRefinerPass

_log = logging.getLogger(__name__)

#: Distilled schedule of the LTX-2.5 alpha-gen workflow.
_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"

#: Nodes the graph needs (core ComfyUI + ComfyUI-VideoHelperSuite).
_NODES = (
    "UNETLoader",
    "LTXICLoRALoaderModelOnly",
    "LTXAddVideoICLoRAGuide",
    "LTXVConditioning",
    "LTXVEmptyLatentAudio",
    "LTXVConcatAVLatent",
    "LTXVSeparateAVLatent",
    "LTXVCropGuides",
)

#: LTX-2.5 transformer / text encoder as GGUF (ComfyUI-GGUF), picked when no
#: safetensors of the same model is on the server; best quality first. The
#: key is a set of name tokens, all of which must appear (case-insensitive):
#: community GGUFs drop the "22b" ("LTX-2.5-Distilled-Q4_K_S.gguf").
_GGUF_UNET = (("ltx-2.5", "distilled"), (".gguf",))
_GGUF_TE = (("gemma4-12b-with-proj-ltx-2.5",), (".gguf",))
_GGUF_ORDER = ("Q8_0", "Q6_K", "Q5_K", "Q4_K", "Q3_K", "Q2_K")
#: Another precision of the configured safetensors (same model) is preferred
#: over any GGUF: int8, then fp8, then bf16 (all fit a 32 GB card with offload).
_PRECISION_ORDER = ("int8", "fp8", "bf16", "fp16")
_PRECISION_TOKENS = (
    "-comfy-int8-convrot",
    "-int8",
    "_int8",
    "-fp8",
    "_fp8",
    "-bf16",
    "_bf16",
    "-fp16",
    "_fp16",
)

_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}


class LTXAlphaRefinerPass(RVMRefinerPass):
    name = "ltx_alpha_refiner"
    version = "0.1.0"
    license = License(
        spdx="LicenseRef-LTX-2-Community",
        commercial_use=True,
        commercial_tool_resale=False,
        notes=(
            "LTX-2.5 + alpha-gen IC-LoRA (Lightricks), run in the user's own "
            "ComfyUI. LTX-2 Community License: free for organisations under "
            "USD 10M annual revenue; above that a commercial licence from "
            "Lightricks is required. Weights are not bundled."
        ),
    )

    # Video model, temporally coherent by itself - but cross-faded chunks
    # can still breathe a little: let the smoother see it like the others.
    smoothable_channels = [CH_MATTE_R, CH_MATTE_G, CH_MATTE_B, CH_MATTE_A]

    DEFAULT_PARAMS: dict[str, Any] = {
        "comfy_url": "http://127.0.0.1:8188",
        "unet": "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
        "lora": "LTX-2.5\\ltx-2.5-22b-ic-lora-alpha-gen-0.9.safetensors",
        "video_vae": "ltx-2.5-video-vae-bf16.safetensors",
        "audio_vae": "ltx-2.5-audio-vae-bf16.safetensors",
        "text_encoder": "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
        "seed": 42,
        "chunk": 49,  # frames per LTX call, 8n+1
        "overlap": 16,  # cross-faded frames between chunks
        "long_edge": 1280,  # working resolution of the crop
        "pad": 0.25,  # box padding, fraction of the box size
        # A neighbour whose box is at most this times the object's PADDED
        # box is kept whole inside the crop; bigger ones (car, wall) may be
        # cut. Scenery neighbours (sky, building, tree, ...) never grow it.
        "neighbour_max_ratio": 1.5,
        # "dilate": SAM mask grown by `gate_frac` of the long edge, feathered.
        # "trimap": ViTMatte's guided trimap (SAM + BiRefNet): wide only
        # where BiRefNet is unsure (hair, blur), tight where it is sure of
        # background (a car door, a window frame touching the subject).
        "gate": "trimap",
        "gate_frac": 0.03,  # SAM gate dilation, fraction of the long edge
        "timeout_s": 900,  # per chunk
        # How the crops reach ComfyUI: "auto" = a temp folder when ComfyUI is
        # on this machine, an upload when it is remote; or "path" / "upload".
        "send_frames": "auto",
        # Dev: save each track's ungated LTX alpha + SAM mask here (npz).
        "debug_dir": "",
        # "whole_frame": one LTX run on the full frame, split between the
        # objects by nearest track (see the module docstring).
        # "per_object": one LTX run per object on a crop, gated by SAM.
        "mode": "whole_frame",
        # Object names refined on their own crop even in whole-frame mode.
        "crop_labels": [],
        # Whole frame: LTX decides what the foreground is, and it is mostly
        # people. A tracked object it does not see as foreground (a parked
        # car, a prop, scenery in a hero slot) would get an empty matte: when
        # an object's share covers less than this fraction of its SAM mask,
        # the object is matted without LTX (ViTMatte for a subject, SAM's
        # edge anti-aliased for scenery) and the log says so. 0 = never.
        "whole_min_cover": 0.5,
        "whole_long_edge": 1920,  # whole-frame working resolution
        "whole_chunk": 97,  # whole-frame frames per LTX call, 8n+1
        # A clip only a few frames longer than `whole_chunk` goes in ONE call
        # of up to this many frames instead of two almost identical ones.
        "whole_chunk_max": 121,
        "far_frac": 0.06,  # foreground farther than this x width from every track: no owner
        "seam_px": 2.0,  # feather of the seam between two objects (working-res px)
        # Folder for the raw LTX results (one .npy per call); "" = no cache.
        "cache_dir": "",
    }

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self._fallback: ViTMatteRefinerPass | None = None
        self._comfy_ok: bool | None = None
        self._models: dict[str, tuple[str, str]] = {}
        self._track_id: int | None = None
        self._track_label = ""
        self._track_scenery = False
        self._first = 0
        # Whole-frame mode: LTX foreground at working res (T, wh, ww) and the
        # nearest-owner map (T, wh, ww), -1 = no owner; built once per shot.
        self._fg_work: np.ndarray | None = None
        self._owner: np.ndarray | None = None
        self._owner_ids: list[int] = []
        self._plate_hw = (0, 0)

    def _trust_outline(self) -> bool:
        # Generates its own outline on purpose; the SAM gate bounds it.
        return True

    def _load_model(self) -> None:
        # Nothing local: the model lives in ComfyUI.
        self._model = object()

    # ------------------------------------------------------------------
    # Hooks: remember which track / which frame range is being refined
    # ------------------------------------------------------------------

    def run_shot(
        self, reader: Any, frame_range: tuple[int, int]
    ) -> dict[int, dict[str, np.ndarray]]:
        self._first = int(frame_range[0])
        self._reset_whole_frame()
        try:
            out = super().run_shot(reader, frame_range)
        finally:
            if self._comfy_ok:
                self._free()
        if self._fg_work is not None and bool(self.params.get("pack_heroes", True)):
            # The whole foreground, owned or not (props, blood, extras).
            import cv2

            sfx = str(self.params.get("channel_suffix", "") or "")
            h, w = self._plate_hw
            for i, f in enumerate(range(frame_range[0], frame_range[1] + 1)):
                out.setdefault(f, {})[f"foreground{sfx}.a"] = np.clip(
                    cv2.resize(
                        self._fg_work[i].astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR
                    ),
                    0.0,
                    1.0,
                )
        return out

    def _refine_track(
        self,
        track: dict[str, Any],
        frames: np.ndarray,
        frame_to_local: dict[int, int],
        n_frames: int,
        plate_h: int,
        plate_w: int,
        track_id: int,
    ) -> tuple[np.ndarray, list[int]]:
        self._track_id = int(track_id)
        self._track_label = str(track.get("label", "") or "")
        self._track_scenery = _is_scenery_track(track)
        return super()._refine_track(
            track, frames, frame_to_local, n_frames, plate_h, plate_w, track_id
        )

    # ------------------------------------------------------------------
    # Per track
    # ------------------------------------------------------------------

    def _skip_refine(self, track_id: int, track: dict[str, Any]) -> bool:
        # Scenery has no hair / motion-blur edge to recover, and LTX on a
        # frame-sized crop of sky returns whatever is salient in it (the car).
        if not _is_scenery_track(track):
            return False
        _log.info(
            "LTX alpha: track %d '%s' is scenery - SAM mask kept (anti-aliased), no LTX run.",
            int(track_id),
            str(track.get("label", "") or ""),
        )
        return True

    def _refine_instance(self, plate_stack: np.ndarray, hard_stack: np.ndarray) -> np.ndarray:
        if not self._comfy_ready():
            return self._vitmatte()._refine_instance(plate_stack, hard_stack)
        if not self._on_crop():
            share = self._whole_frame_share(plate_stack, hard_stack)
            return self._unless_unseen(share, plate_stack, hard_stack)
        import cv2

        T, H, W = hard_stack.shape
        present = [t for t in range(T) if hard_stack[t].any()]
        if not present:
            return np.zeros((T, H, W), np.float32)
        a, b = present[0], present[-1] + 1  # the object's span
        chunk = int(self.params["chunk"])
        overlap = int(self.params["overlap"])
        acc = np.zeros((T, H, W), np.float32)
        # Per PIXEL: two chunks' boxes differ, so a pixel outside one box
        # must not be averaged with that chunk's zero.
        wsum = np.zeros((T, H, W), np.float32)
        work = make_work_dir("laov_ltx_")
        try:
            spans = chunk_spans(b - a, chunk, overlap)
            for ci, (s0, nf) in enumerate(spans):
                # Frames of this call (a short object: the plate around it,
                # then the last frame repeated up to LTX's 8n+1 length).
                idx = [min(a + s0 + j, T - 1) for j in range(nf)]
                own = [k for k in dict.fromkeys(idx) if k < b]
                box = self._chunk_box(hard_stack, own, H, W)
                if box is None:
                    continue
                y0, x0, y1, x1 = box
                ww, wh = work_size(x1 - x0, y1 - y0, int(self.params["long_edge"]))
                d = work / f"c{ci:03d}"
                d.mkdir()
                for j, k in enumerate(idx):
                    crop = cv2.resize(
                        np.clip(plate_stack[k, y0:y1, x0:x1, :3], 0.0, 1.0),
                        (ww, wh),
                        interpolation=cv2.INTER_AREA if ww < x1 - x0 else cv2.INTER_CUBIC,
                    )
                    cv2.imwrite(
                        str(d / f"f_{j:04d}.png"),
                        cv2.cvtColor((crop * 255.0 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR),
                    )
                t0 = time.time()
                raw = self._ltx(d, ww, wh, nf)
                _log.info(
                    "LTX alpha: track %s chunk %d/%d, frames %d-%d, box %dx%d -> %dx%d, %.1fs",
                    self._track_id,
                    ci + 1,
                    len(spans),
                    self._first + idx[0],
                    self._first + idx[-1],
                    x1 - x0,
                    y1 - y0,
                    ww,
                    wh,
                    time.time() - t0,
                )
                raw = levels(raw)
                back = cv2.INTER_AREA if ww > x1 - x0 else cv2.INTER_LINEAR
                for j, k in enumerate(idx):
                    if k >= b or (j > 0 and idx[j - 1] == k):
                        continue  # padding frames
                    w = crossfade_weight(j, nf, overlap, ci == 0, ci == len(spans) - 1)
                    acc[k, y0:y1, x0:x1] += w * cv2.resize(
                        raw[j], (x1 - x0, y1 - y0), interpolation=back
                    )
                    wsum[k, y0:y1, x0:x1] += w
        finally:
            shutil.rmtree(work, ignore_errors=True)
        alpha = acc / np.maximum(wsum, 1e-6)
        if self.params.get("debug_dir"):
            d = Path(str(self.params["debug_dir"]))
            d.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                d / f"ltx_track{self._track_id}.npz",
                alpha=alpha.astype(np.float16),
                hard=(hard_stack > 0).astype(np.uint8),
                first=self._first,
            )
        if str(self.params.get("gate", "dilate")) == "trimap":
            return alpha * self._trimap_gate(plate_stack, hard_stack)
        return alpha * sam_gate(hard_stack, float(self.params["gate_frac"]))

    # ------------------------------------------------------------------
    # Whole-frame mode
    # ------------------------------------------------------------------

    def _on_crop(self) -> bool:
        """Is the track being refined matted on its own crop?"""
        if str(self.params.get("mode", "whole_frame")) != "whole_frame":
            return True
        crop = {str(x) for x in (self.params.get("crop_labels") or [])}
        return bool(self._track_label) and self._track_label in crop

    def _unless_unseen(
        self, share: np.ndarray, plate_stack: np.ndarray, hard_stack: np.ndarray
    ) -> np.ndarray:
        """`share`, or a matte made without LTX when LTX did not see this
        object as foreground (`whole_min_cover`)."""
        floor = float(self.params.get("whole_min_cover", 0.5) or 0.0)
        sam_px = float(np.count_nonzero(hard_stack))
        if floor <= 0.0 or sam_px == 0.0:
            return share
        cover = float(np.count_nonzero((share > 0.5) & (hard_stack > 0))) / sam_px
        if cover >= floor:
            return share
        how = "SAM's edge, anti-aliased" if self._track_scenery else "ViTMatte"
        _log.warning(
            "LTX alpha: '%s' (track %s) is not foreground for LTX (its share covers %.0f%% "
            "of its SAM mask): %s for this object.",
            self._track_label or "object",
            self._track_id,
            100.0 * cover,
            how,
        )
        self.unseen.append(self._track_label or str(self._track_id))
        if self._track_scenery:
            return antialiased(hard_stack)
        return self._vitmatte()._refine_instance(plate_stack, hard_stack)

    def _reset_whole_frame(self) -> None:
        self._fg_work, self._owner, self._owner_ids = None, None, []
        #: Objects of the last run that LTX did not see as foreground.
        self.unseen: list[str] = []

    def _whole_frame_share(self, plate_stack: np.ndarray, hard_stack: np.ndarray) -> np.ndarray:
        """This track's share of the whole-frame LTX foreground."""
        import cv2

        T, H, W = hard_stack.shape
        if self._fg_work is None:
            self._plate_hw = (H, W)
            self._fg_work = self._whole_frame_alpha(plate_stack)
            self._owner, self._owner_ids = self._owner_map(T)
        assert self._owner is not None
        if self._track_id not in self._owner_ids:
            # A part refined on its own (parts_from_whole off): no owner of
            # its own, so the SAM gate decides, as in per-object mode.
            up = np.stack(
                [
                    cv2.resize(
                        self._fg_work[t].astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR
                    )
                    for t in range(T)
                ]
            )
            return up * sam_gate(hard_stack, float(self.params["gate_frac"]))
        k = self._owner_ids.index(int(self._track_id))
        seam = float(self.params["seam_px"])
        out = np.zeros((T, H, W), np.float32)
        for t in range(T):
            if not hard_stack[t].any():
                continue
            mine = (self._owner[t] == k).astype(np.float32)
            share = cv2.GaussianBlur(mine, (0, 0), seam) if seam > 0 else mine
            # Sum of the blurred one-hots = blur(has an owner) <= 1, so the
            # objects' shares add up to the foreground away from the seams.
            out[t] = cv2.resize(
                share * self._fg_work[t].astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR
            )
        return np.clip(out, 0.0, 1.0)

    def _whole_frame_alpha(self, plate_stack: np.ndarray) -> np.ndarray:
        """LTX on the full frame, chunked and cross-faded: (T, wh, ww) float16."""
        import cv2

        T, H, W = plate_stack.shape[:3]
        ww, wh = work_size(W, H, int(self.params["whole_long_edge"]))
        chunk = single_call_chunk(
            T, int(self.params["whole_chunk"]), int(self.params.get("whole_chunk_max", 0))
        )
        overlap = int(self.params["overlap"])
        acc = np.zeros((T, wh, ww), np.float32)
        wsum = np.zeros((T, 1, 1), np.float32)
        work = make_work_dir("laov_ltx_wf_")
        try:
            spans = chunk_spans(T, chunk, overlap)
            for ci, (s0, nf) in enumerate(spans):
                idx = [min(s0 + j, T - 1) for j in range(nf)]
                d = work / f"c{ci:03d}"
                d.mkdir()
                for j, k in enumerate(idx):
                    fr = cv2.resize(
                        np.clip(plate_stack[k, :, :, :3], 0.0, 1.0),
                        (ww, wh),
                        interpolation=cv2.INTER_AREA if ww < W else cv2.INTER_CUBIC,
                    )
                    cv2.imwrite(
                        str(d / f"f_{j:04d}.png"),
                        cv2.cvtColor((fr * 255.0 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR),
                    )
                t0 = time.time()
                raw = levels(self._ltx(d, ww, wh, nf))  # each call levelled on its own
                _log.info(
                    "LTX alpha (whole frame): chunk %d/%d, frames %d-%d at %dx%d, %.1fs",
                    ci + 1,
                    len(spans),
                    self._first + idx[0],
                    self._first + idx[-1],
                    ww,
                    wh,
                    time.time() - t0,
                )
                for j, k in enumerate(idx):
                    if j > 0 and idx[j - 1] == k:
                        continue  # padding frames
                    w = crossfade_weight(j, nf, overlap, ci == 0, ci == len(spans) - 1)
                    acc[k] += w * raw[j]
                    wsum[k] += w
                shutil.rmtree(d, ignore_errors=True)
        finally:
            shutil.rmtree(work, ignore_errors=True)
        return (acc / np.maximum(wsum, 1e-6)).astype(np.float16)

    def _owner_map(self, T: int) -> tuple[np.ndarray, list[int]]:
        """Nearest track per working-res pixel (-1 = none within `far_frac`).
        Owners = every top-level SUBJECT track (parts belong to their whole),
        whether or not this pass refines it: a neighbour refined by another
        engine still keeps its foreground. Scenery (sky, building, tree) owns
        nothing: LTX's foreground is by definition not the set, and a wall
        tracked behind an actor would otherwise take the hair that sticks out
        of the actor's SAM mask (it lies inside the wall's mask)."""
        import cv2

        assert self._fg_work is not None
        wh, ww = self._fg_work.shape[1:]
        ids = [
            int(t)
            for t, tr in sorted(self._hard_masks.items())
            if (tr.get("parent") is None or int(tr["parent"]) not in self._hard_masks)
            and not _is_scenery_track(tr)
        ]
        far = float(self.params["far_frac"]) * ww
        owner = np.full((T, wh, ww), -1, np.int16)
        dense: list[dict[int, np.ndarray]] = []
        for t in ids:
            tr = self._hard_masks[t]
            st = tr.get("stack")
            per: dict[int, np.ndarray] = {}
            if st is None:
                continue
            for k, f in enumerate(tr.get("frames") or []):
                li = int(f) - self._first
                if 0 <= li < T and np.asarray(st[k]).any():
                    per[li] = cv2.resize(
                        (np.asarray(st[k]) > 0).astype(np.uint8),
                        (ww, wh),
                        interpolation=cv2.INTER_NEAREST,
                    )
            dense.append(per)
        for i in range(T):
            ds = [
                cv2.distanceTransform(1 - per[i], cv2.DIST_L2, 5)
                if i in per and per[i].any()
                else np.full((wh, ww), np.inf, np.float32)
                for per in dense
            ]
            if not ds:
                continue
            d = np.stack(ds)
            near = np.argmin(d, 0)
            dmin = np.take_along_axis(d, near[None], 0)[0]
            owner[i] = np.where(dmin <= far, near, -1).astype(np.int16)
        return owner, ids

    def _trimap_gate(self, plate_stack: np.ndarray, hard_stack: np.ndarray) -> np.ndarray:
        """1 where ViTMatte's guided trimap is not certain background,
        feathered by a couple of pixels."""
        import cv2

        from live_action_aov.passes.matte.vitmatte import guided_trimap

        vit = self._vitmatte()
        out = np.zeros(hard_stack.shape, np.float32)
        for t in range(hard_stack.shape[0]):
            m = (hard_stack[t] > 0).astype(np.uint8)
            if not m.any():
                continue
            tri = guided_trimap(m, vit._guide_alpha(plate_stack[t], m))
            out[t] = cv2.GaussianBlur((tri > 0).astype(np.float32), (0, 0), 2.0)
        return out

    def _chunk_box(
        self, hard_stack: np.ndarray, frames: list[int], H: int, W: int
    ) -> tuple[int, int, int, int] | None:
        m = np.any(hard_stack[frames] > 0, axis=0)
        if not m.any():
            return None
        ys, xs = np.nonzero(m)
        own = (int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1)
        return grow_box(
            own,
            self._neighbour_boxes(frames),
            float(self.params["pad"]),
            float(self.params["neighbour_max_ratio"]),
            H,
            W,
        )

    def _neighbour_boxes(self, local_frames: list[int]) -> list[tuple[int, int, int, int]]:
        """Union box over the chunk of every OTHER subject track (parts and
        the whole of this object excluded: they are the object itself;
        scenery excluded: it must never grow the crop)."""
        me = self._track_id
        family = {me}
        for t, tr in self._hard_masks.items():
            p = tr.get("parent")
            if p is not None and (int(t) == me or int(p) == me):
                family |= {int(t), int(p)}
        want = {self._first + k for k in local_frames}
        out = []
        for t, tr in self._hard_masks.items():
            if int(t) in family or _is_scenery_track(tr):
                continue
            boxes = []
            st = tr.get("stack")
            for k, f in enumerate(tr.get("frames") or []):
                if int(f) in want:
                    bm = stack_box(st, k)
                    if bm.crop.size and bm.crop.any():
                        boxes.append(bm.box)
            if boxes:
                out.append(
                    (
                        min(b[0] for b in boxes),
                        min(b[1] for b in boxes),
                        max(b[2] for b in boxes),
                        max(b[3] for b in boxes),
                    )
                )
        return out

    # ------------------------------------------------------------------
    # ComfyUI (tests override `_comfy_ready` / `_run_ltx`)
    # ------------------------------------------------------------------

    def _url(self, path: str) -> str:
        return str(self.params["comfy_url"]).rstrip("/") + path

    def _comfy_ready(self) -> bool:
        if self._comfy_ok is None:
            self._comfy_ok, msg, self._models = _resolve(self.params)
            if self._comfy_ok:
                _log.info("LTX alpha: %s", msg)
            else:
                _log.warning("LTX alpha: %s - using ViTMatte for these objects.", msg)
        return self._comfy_ok

    def _vitmatte(self) -> ViTMatteRefinerPass:
        if self._fallback is None:
            from live_action_aov.passes.matte.vitmatte import ViTMatteRefinerPass

            self._fallback = ViTMatteRefinerPass({})
        return self._fallback

    def _upload(self) -> bool:
        mode = str(self.params.get("send_frames", "auto"))
        if mode != "auto":
            return mode == "upload"
        host = urllib.parse.urlparse(str(self.params["comfy_url"])).hostname or ""
        return host.lower() not in _LOCAL_HOSTS

    def _send_frames(self, folder: Path) -> dict[str, Any]:
        """The image-loading node: the folder itself when ComfyUI shares this
        disk, else the PNGs uploaded into a fresh subfolder of its input/."""
        common = {"image_load_cap": 0, "skip_first_images": 0, "select_every_nth": 1}
        if not self._upload():
            return {
                "class_type": "VHS_LoadImagesPath",
                "inputs": {"directory": str(folder), **common},
            }
        sub = f"laov_ltx_{uuid.uuid4().hex[:10]}"
        if not getattr(self, "_upload_warned", False):
            # ComfyUI has no delete endpoint: the crops stay on that machine.
            _log.warning(
                "LTX alpha: plate crops uploaded to the remote ComfyUI's input/laov_ltx_* "
                "folders; ComfyUI cannot delete them remotely - clear them on that machine."
            )
            self._upload_warned = True
        for f in sorted(folder.glob("*.png")):
            upload_image(self._url("/upload/image"), f, sub)
        return {"class_type": "VHS_LoadImages", "inputs": {"directory": sub, **common}}

    def _name(self, key: str) -> str:
        """The server's file name for a configured model (resolved by `_resolve`)."""
        got = self._models.get(key)
        return got[1] if got else str(self.params[key])

    def _graph(self, folder: Path, ww: int, wh: int, nf: int) -> dict[str, Any]:
        p = self.params
        unet_node, unet = self._models.get("unet", ("UNETLoader", str(p["unet"])))
        te_node, te = self._models.get("text_encoder", ("CLIPLoader", str(p["text_encoder"])))
        unet_inputs: dict[str, Any] = {"unet_name": unet}
        if unet_node == "UNETLoader":
            unet_inputs["weight_dtype"] = "default"
        return {
            "1": {"class_type": unet_node, "inputs": unet_inputs},
            "2": {
                "class_type": "LTXICLoRALoaderModelOnly",
                "inputs": {
                    "model": ["1", 0],
                    "lora_name": self._name("lora"),
                    "strength_model": 1.0,
                },
            },
            "3": {"class_type": "VAELoader", "inputs": {"vae_name": self._name("video_vae")}},
            "4": {"class_type": "VAELoader", "inputs": {"vae_name": self._name("audio_vae")}},
            "5": {"class_type": te_node, "inputs": {"clip_name": te, "type": "ltxv"}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["5", 0]}},
            "7": {
                "class_type": "LTXVConditioning",
                "inputs": {"positive": ["6", 0], "negative": ["6", 0], "frame_rate": 24.0},
            },
            "8": self._send_frames(folder),
            "9": {
                "class_type": "EmptyLTXVLatentVideo",
                "inputs": {"width": ww, "height": wh, "length": nf, "batch_size": 1},
            },
            "10": {
                "class_type": "LTXAddVideoICLoRAGuide",
                "inputs": {
                    "positive": ["7", 0],
                    "negative": ["7", 1],
                    "vae": ["3", 0],
                    "latent": ["9", 0],
                    "image": ["8", 0],
                    "frame_idx": 0,
                    "strength": 1.0,
                    "latent_downscale_factor": ["2", 1],
                    "crop": "disabled",
                    "use_tiled_encode": False,
                    "tile_size": 256,
                    "tile_overlap": 64,
                },
            },
            "11": {
                "class_type": "LTXVEmptyLatentAudio",
                "inputs": {
                    "frames_number": nf,
                    "frame_rate": 24.0,
                    "batch_size": 1,
                    "audio_vae": ["4", 0],
                },
            },
            "12": {
                "class_type": "LTXVConcatAVLatent",
                "inputs": {"video_latent": ["10", 2], "audio_latent": ["11", 0]},
            },
            "13": {"class_type": "RandomNoise", "inputs": {"noise_seed": int(p["seed"])}},
            "14": {
                "class_type": "CFGGuider",
                "inputs": {
                    "model": ["2", 0],
                    "positive": ["10", 0],
                    "negative": ["10", 1],
                    "cfg": 1.0,
                },
            },
            "15": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler_ancestral"}},
            "16": {"class_type": "ManualSigmas", "inputs": {"sigmas": _SIGMAS}},
            "17": {
                "class_type": "SamplerCustomAdvanced",
                "inputs": {
                    "noise": ["13", 0],
                    "guider": ["14", 0],
                    "sampler": ["15", 0],
                    "sigmas": ["16", 0],
                    "latent_image": ["12", 0],
                },
            },
            "18": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["17", 0]}},
            "19": {
                "class_type": "LTXVCropGuides",
                "inputs": {"positive": ["10", 0], "negative": ["10", 1], "latent": ["18", 0]},
            },
            "20": {
                "class_type": "VAEDecodeTiled",
                "inputs": {
                    "samples": ["19", 2],
                    "vae": ["3", 0],
                    "tile_size": 512,
                    "overlap": 64,
                    "temporal_size": 64,
                    "temporal_overlap": 8,
                },
            },
            # PreviewImage, not SaveImage: the mattes land in ComfyUI's temp
            # folder (wiped when ComfyUI starts and exits), not in its
            # output/ where client-derived frames would pile up.
            "21": {"class_type": "PreviewImage", "inputs": {"images": ["20", 0]}},
        }

    def _ltx(self, folder: Path, ww: int, wh: int, nf: int) -> np.ndarray:
        """`_run_ltx` through the `cache_dir` cache (key: the frames sent +
        the models and settings that change the result)."""
        cache = str(self.params.get("cache_dir", "") or "")
        if not cache:
            return self._run_ltx(folder, ww, wh, nf)
        key = ltx_cache_key(folder, ww, wh, nf, self.params, self._models)
        f = Path(cache) / f"ltx_{key}.npy"
        if f.is_file():
            try:
                raw = np.load(f)
                if raw.shape[0] >= nf and raw.shape[1:] == (wh, ww):
                    _log.info("LTX alpha: reused cached call %s", f.name)
                    return raw[:nf].astype(np.float32)
            except (OSError, ValueError):
                pass  # unreadable: run it again
        raw = self._run_ltx(folder, ww, wh, nf)
        try:
            f.parent.mkdir(parents=True, exist_ok=True)
            tmp = f.with_name(f.stem + "_tmp.npy")
            np.save(tmp, raw.astype(np.float16))
            tmp.replace(f)
        except OSError as e:
            _log.warning("LTX alpha: could not cache the call (%s)", e)
        return raw

    def _run_ltx(self, folder: Path, ww: int, wh: int, nf: int) -> np.ndarray:
        """One LTX call: the PNG crops in `folder` -> (nf, wh, ww) alpha."""
        import cv2

        body = json.dumps(
            {"prompt": self._graph(folder, ww, wh, nf), "client_id": "liveactionaov"}
        ).encode()
        req = urllib.request.Request(
            self._url("/prompt"), data=body, headers={"Content-Type": "application/json"}
        )
        try:
            pid = json.load(urllib.request.urlopen(req, timeout=60))["prompt_id"]
        except urllib.error.HTTPError as e:
            raise RuntimeError(
                f"LTX alpha: ComfyUI refused the graph: {e.read().decode()[:1500]}"
            ) from e
        deadline = time.time() + float(self.params["timeout_s"])
        while True:
            hist = json.load(urllib.request.urlopen(self._url(f"/history/{pid}"), timeout=60))
            if pid in hist:
                st = hist[pid].get("status", {})
                if st.get("status_str") == "error":
                    raise RuntimeError(f"LTX alpha: ComfyUI error: {json.dumps(st)[:1500]}")
                if st.get("completed"):
                    break
            if time.time() > deadline:
                raise RuntimeError(f"LTX alpha: ComfyUI took over {self.params['timeout_s']} s")
            time.sleep(1.0)
        out = []
        for im in hist[pid]["outputs"]["21"]["images"]:
            q = urllib.parse.urlencode(
                {"filename": im["filename"], "subfolder": im["subfolder"], "type": im["type"]}
            )
            data = urllib.request.urlopen(self._url(f"/view?{q}"), timeout=60).read()
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            out.append(img.astype(np.float32).mean(axis=2) / 255.0)
        return np.stack(out)[:nf]

    def _free(self) -> None:
        """Unload LTX from ComfyUI's VRAM (the server keeps running)."""
        try:
            urllib.request.urlopen(
                urllib.request.Request(
                    self._url("/free"),
                    data=json.dumps({"unload_models": True, "free_memory": True}).encode(),
                    headers={"Content-Type": "application/json"},
                ),
                timeout=30,
            )
        except (urllib.error.URLError, OSError):
            pass


# ----------------------------------------------------------------------
# Pure helpers (unit-tested)
# ----------------------------------------------------------------------


def comfy_check(params: dict[str, Any] | None = None) -> tuple[bool, str]:
    """Is the ComfyUI at `comfy_url` ready for LTX Alpha? (ok, one-line
    message for the user). Checks the server, the nodes and the model files."""
    ok, msg, _models = _resolve(params)
    return ok, msg


def _why_unreachable(e: BaseException) -> str:
    """A connection error in words an artist can act on."""
    text = str(e)
    reason = getattr(e, "reason", e)
    if isinstance(reason, ConnectionRefusedError) or "10061" in text or "refused" in text.lower():
        return (
            "nothing is running there. Start ComfyUI, or check the port in the address "
            "(8188 unless you changed it). Until then the mattes use ViTMatte."
        )
    if isinstance(reason, TimeoutError) or "timed out" in text.lower():
        return "no answer in 20 s. Is that machine on, and the port open?"
    return text


def _resolve(
    params: dict[str, Any] | None,
) -> tuple[bool, str, dict[str, tuple[str, str]]]:
    """`comfy_check` + which loader node / file serves the transformer and
    the text encoder on this server (int8 safetensors first, else GGUF)."""
    p = {**LTXAlphaRefinerPass.DEFAULT_PARAMS, **(params or {})}
    url = str(p["comfy_url"]).rstrip("/")
    try:
        info = json.load(urllib.request.urlopen(url + "/object_info", timeout=20))
    except (urllib.error.URLError, OSError, ValueError) as e:
        return False, f"ComfyUI not reachable at {url}: {_why_unreachable(e)}", {}
    picked = {
        "unet": pick_model(
            info, str(p["unet"]), "UNETLoader", "UnetLoaderGGUF", "unet_name", _GGUF_UNET
        ),
        "text_encoder": pick_model(
            info, str(p["text_encoder"]), "CLIPLoader", "CLIPLoaderGGUF", "clip_name", _GGUF_TE
        ),
    }
    missing = [f"node {n}" for n in _NODES if n not in info]
    if not any(n in info for n in ("VHS_LoadImagesPath", "VHS_LoadImages")):
        missing.append("node VHS_LoadImages (ComfyUI-VideoHelperSuite)")
    missing += [f"{p[k]} (or a GGUF of it)" for k, v in picked.items() if v is None]
    for key, node, field in (
        ("lora", "LTXICLoRALoaderModelOnly", "lora_name"),
        ("video_vae", "VAELoader", "vae_name"),
        ("audio_vae", "VAELoader", "vae_name"),
    ):
        if node not in info:
            continue
        choices = _choices(info[node], field)
        name = str(p[key]) if str(p[key]) in choices else find_by_basename(choices, str(p[key]))
        if name is None:
            missing.append(str(p[key]))
        else:
            picked[key] = (node, name)
    if missing:
        return False, f"ComfyUI at {url} is missing: " + ", ".join(missing), {}
    models = {k: v for k, v in picked.items() if v is not None}
    quant = [Path(v[1]).name for v in models.values() if v[1].lower().endswith(".gguf")]
    extra = f", GGUF: {', '.join(quant)}" if quant else ""
    return True, f"ComfyUI ready at {url} (LTX-2.5 + alpha-gen LoRA found{extra})", models


def pick_model(
    info: dict[str, Any],
    wanted: str,
    node: str,
    gguf_node: str,
    field: str,
    gguf: tuple[Any, tuple[str, ...]],
) -> tuple[str, str] | None:
    """(loader node, file) for a model, None if the server has none of these:

    1. `wanted` exactly;
    2. the same file in another folder of the model tree (studios keep LoRAs and
       models in sub-folders: `ltxv/ltx25/...` instead of `LTX-2.5/...`);
    3. another precision of the same safetensors (int8 > fp8 > bf16), e.g. the
       bf16 distilled transformer when the int8 one was never downloaded;
    4. the best-quality GGUF whose name holds every token of gguf[0].
    """
    if node in info:
        choices = _choices(info[node], field)
        if wanted in choices:
            return node, wanted
        same = find_by_basename(choices, wanted)
        if same:
            return node, same
        other = _other_precision(choices, wanted)
        if other:
            return node, other
    if gguf_node not in info:
        return None
    key, exts = gguf
    tokens = (key,) if isinstance(key, str) else tuple(key)
    cands = [
        f
        for f in _choices(info[gguf_node], field)
        if all(t.lower() in f.lower() for t in tokens) and f.lower().endswith(exts)
    ]

    def rank(f: str) -> int:
        u = f.upper()
        return next((i for i, q in enumerate(_GGUF_ORDER) if q in u), len(_GGUF_ORDER))

    return (gguf_node, sorted(cands, key=rank)[0]) if cands else None


def _basename(name: str) -> str:
    return name.replace("\\", "/").rsplit("/", 1)[-1].lower()


def find_by_basename(choices: list[str], wanted: str) -> str | None:
    """The server's name for `wanted` when it sits in another sub-folder."""
    b = _basename(wanted)
    return next((c for c in choices if _basename(c) == b), None)


def _strip_precision(name: str) -> str:
    n = _basename(name)
    for t in _PRECISION_TOKENS:
        n = n.replace(t, "")
    return n


def _other_precision(choices: list[str], wanted: str) -> str | None:
    """Same model file in another precision (int8 > fp8 > bf16)."""
    stem = _strip_precision(wanted)
    same = [
        c for c in choices if _strip_precision(c) == stem and c.lower().endswith(".safetensors")
    ]

    def rank(c: str) -> int:
        b = _basename(c)
        return next((i for i, q in enumerate(_PRECISION_ORDER) if q in b), len(_PRECISION_ORDER))

    return sorted(same, key=rank)[0] if same else None


def upload_image(url: str, path: Path, subfolder: str) -> None:
    """POST one PNG to ComfyUI's /upload/image, into input/<subfolder>/."""
    boundary = uuid.uuid4().hex
    body = b""
    for name, value in (("subfolder", subfolder), ("type", "input"), ("overwrite", "true")):
        body += (
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'
        ).encode()
    body += (
        f"--{boundary}\r\nContent-Disposition: form-data; "
        f'name="image"; filename="{path.name}"\r\nContent-Type: image/png\r\n\r\n'
    ).encode()
    body += path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    urllib.request.urlopen(req, timeout=120).read()


def _choices(node_info: dict[str, Any], field: str) -> list[str]:
    """A combo input's options: `[[...]]` (classic) or `["COMBO", {"options": [...]}]`."""
    spec = (node_info.get("input", {}).get("required", {}) or {}).get(field)
    if not spec:
        return []
    first = spec[0]
    if isinstance(first, list):
        return [str(x) for x in first]
    if len(spec) > 1 and isinstance(spec[1], dict):
        return [str(x) for x in spec[1].get("options") or []]
    return []


def ltx_cache_key(
    folder: Path,
    ww: int,
    wh: int,
    nf: int,
    params: dict[str, Any],
    models: dict[str, tuple[str, str]],
) -> str:
    """Hash of everything that changes an LTX call's result: the PNG frames
    sent, the working size, the seed and the model files used."""
    h = hashlib.sha1()
    keep = ("seed", "unet", "lora", "video_vae", "audio_vae", "text_encoder")
    h.update(
        json.dumps(
            {
                "ww": ww,
                "wh": wh,
                "nf": nf,
                "params": {k: str(params.get(k, "")) for k in keep},
                "models": {k: list(v) for k, v in sorted(models.items())},
                "sigmas": _SIGMAS,
            },
            sort_keys=True,
        ).encode()
    )
    for f in sorted(folder.glob("*.png")):
        h.update(f.read_bytes())
    return h.hexdigest()[:20]


def chunk_spans(n: int, chunk: int, overlap: int) -> list[tuple[int, int]]:
    """(start, length) of each LTX call over `n` frames. Lengths are 8k+1;
    a span shorter than `chunk` is one call rounded UP to 8k+1 (the caller
    pads); the last chunk is pinned to the end so nothing is left over."""
    if n <= chunk:
        return [(0, 8 * ((max(n, 1) - 1 + 7) // 8) + 1)]
    step = max(1, chunk - overlap)
    starts = list(range(0, n - chunk + 1, step))
    if starts[-1] != n - chunk:
        starts.append(n - chunk)
    return [(s, chunk) for s in starts]


def antialiased(hard_stack: np.ndarray) -> np.ndarray:
    """(T, H, W) hard masks -> float mattes: solid inside, half a pixel of
    soft edge outside."""
    import cv2

    out = np.zeros(hard_stack.shape, np.float32)
    for t in range(hard_stack.shape[0]):
        if hard_stack[t].any():
            hard = (hard_stack[t] > 0).astype(np.float32)
            out[t] = np.maximum(cv2.GaussianBlur(hard, (0, 0), 0.6), hard)
    return out


def single_call_chunk(n: int, chunk: int, chunk_max: int) -> int:
    """The chunk length to use for `n` frames: `chunk`, or `n` rounded up to
    8k+1 when that fits `chunk_max` (one call instead of two that overlap
    almost entirely: a 100-frame clip with 97-frame chunks ran LTX twice)."""
    one = 8 * ((max(n, 1) - 1 + 7) // 8) + 1
    return one if chunk < n and one <= chunk_max else chunk


def crossfade_weight(j: int, length: int, overlap: int, first: bool, last: bool) -> float:
    """Weight of frame j of a chunk: ramps up over the overlap at its start
    (unless it is the first chunk) and down at its end (unless the last)."""
    w = 1.0
    if not first:
        w = min(w, (j + 1) / (overlap + 1))
    if not last:
        w = min(w, (length - j) / (overlap + 1))
    return float(w)


def _is_scenery_track(track: dict[str, Any]) -> bool:
    """The SAM pass's `scenery` flag; an older artifact without it falls back
    to the label words (sky, building, tree, ...)."""
    flag = track.get("scenery")
    if flag is not None:
        return bool(flag)
    return is_scenery_label(str(track.get("label", "") or ""))


def grow_box(
    own: tuple[int, int, int, int],
    neighbours: list[tuple[int, int, int, int]],
    pad: float,
    max_ratio: float,
    H: int,
    W: int,
    scenery: Sequence[bool] = (),
) -> tuple[int, int, int, int]:
    """Pad the object's box (y0, x0, y1, x1); then any neighbour of similar
    size it cuts is taken in WHOLE, until nothing changes.

    Never taken in: a neighbour flagged in `scenery` (aligned with
    `neighbours`; missing = not scenery), or one whose box is bigger than
    `max_ratio` times the PADDED own box - the sky, a building, the wall
    behind the subject are scenery, and LTX would matte whatever is salient
    in a crop that holds them."""
    y0, x0, y1, x1 = own
    bh, bw = y1 - y0, x1 - x0
    box = (
        max(0, int(y0 - pad * bh)),
        max(0, int(x0 - pad * bw)),
        min(H, int(y1 + pad * bh) + 1),
        min(W, int(x1 + pad * bw) + 1),
    )
    area = (box[2] - box[0]) * (box[3] - box[1])
    subjects = [n for i, n in enumerate(neighbours) if not (i < len(scenery) and scenery[i])]
    small = [n for n in subjects if (n[2] - n[0]) * (n[3] - n[1]) <= max_ratio * area]
    changed = True
    while changed:
        changed = False
        for n in small:
            overlaps = n[0] < box[2] and n[2] > box[0] and n[1] < box[3] and n[3] > box[1]
            inside = n[0] >= box[0] and n[1] >= box[1] and n[2] <= box[2] and n[3] <= box[3]
            if overlaps and not inside:
                box = (min(box[0], n[0]), min(box[1], n[1]), max(box[2], n[2]), max(box[3], n[3]))
                changed = True
    return box


def work_size(cw: int, ch: int, long_edge: int) -> tuple[int, int]:
    """Crop size scaled to `long_edge`, both sides multiples of 32."""
    s = long_edge / max(cw, ch)
    return max(32, round(cw * s / 32) * 32), max(32, round(ch * s / 32) * 32)


def levels(raw: np.ndarray) -> np.ndarray:
    """Alpha-Gen leaves a grey haze: stretch the 0.5 / 99.5 percentiles to 0 / 1."""
    lo, hi = np.percentile(raw, 0.5), np.percentile(raw, 99.5)
    return np.clip((raw - lo) / max(float(hi - lo), 1e-3), 0.0, 1.0).astype(np.float32)


def sam_gate(hard_stack: np.ndarray, frac: float) -> np.ndarray:
    """Soft gate: each frame's SAM mask dilated by `frac` of the long edge,
    feathered over half that."""
    import cv2

    T, H, W = hard_stack.shape
    r = max(3, int(frac * max(H, W)))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    out = np.zeros((T, H, W), np.float32)
    for t in range(T):
        if hard_stack[t].any():
            g = cv2.dilate((hard_stack[t] > 0).astype(np.float32), ker)
            out[t] = np.clip(cv2.GaussianBlur(g, (0, 0), r / 2), 0.0, 1.0)
    return out


__all__ = ["LTXAlphaRefinerPass", "comfy_check"]
