"""Single-frame SAM 3 mask preview for the click-to-mask workflow.

The batch pass propagates clicks across the whole shot — minutes. This
engine answers the prep-time question "what will my points produce?" by
running SAM 3 on ONE frame (the seed frame, at viewport preview
resolution) and handing back the mask for an overlay. The artist adjusts
points and previews again until the mask reads right, THEN submits.

Implementation notes:
- Reuses the exact session API the batch pass uses (`init_video_session`
  with a single frame + `add_inputs_to_inference_session(input_points=…)`
  + propagate) so the preview is the same model answering the same
  question — not an approximation by a different code path.
- The model loads lazily on the FIRST preview (one-time wait) and stays
  resident for instant re-previews. `unload()` frees the VRAM — the
  MainWindow calls it before Submit so the executor gets the whole GPU.
- All torch/transformers imports happen inside the worker thread; the GUI
  process stays import-light until the feature is actually used.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal


class MaskPreviewWorker(QObject):
    """Async single-frame mask preview. One in-flight request at a time;
    a request issued while busy is dropped (the button re-enables on
    completion, so this is belt-and-braces)."""

    ready = Signal(object)  # np.ndarray (H, W) float32 in [0, 1], image res
    failed = Signal(str)
    status = Signal(str)  # short progress line ("loading SAM 3…", "running…")

    def __init__(self) -> None:
        super().__init__()
        self._pool = QThreadPool.globalInstance()
        self._busy = False
        # Model state lives here (worker-thread populated, main-thread
        # freed via unload) — processor, model, device, dtype.
        self._engine: dict[str, Any] = {}

    def is_busy(self) -> bool:
        return self._busy

    def request(
        self,
        image_rgb: np.ndarray,
        points: list[list[float]],
        labels: list[int],
        box: list[float] | None,
        refine: bool = False,
        model_id: str = "",
    ) -> None:
        """Compute the mask for `image_rgb` (H, W, 3 float32 [0,1]) seeded
        by `points` (image px) / `labels` (1=fg, 0=bg) / optional `box`.

        When `refine` is set, the hard SAM 3 mask is passed through the
        BiRefNet refiner (`model_id`, or the pass default) so the overlay
        shows the SOFT result the submit would produce."""
        if self._busy:
            return
        self._busy = True
        task = _PreviewTask(self, image_rgb, points, labels, box, refine, model_id)
        self._pool.start(task)

    def unload(self) -> None:
        """Free the resident SAM 3 model (called before Submit so the batch
        executor gets the full GPU). Safe to call when nothing is loaded."""
        if not self._engine:
            return
        self._engine.clear()
        try:
            import gc

            gc.collect()
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # Internal — called from the worker thread.
    def _finish(self, mask: np.ndarray | None, error: str | None) -> None:
        self._busy = False
        if error is not None:
            self.failed.emit(error)
        else:
            self.ready.emit(mask)


class _PreviewTask(QRunnable):
    def __init__(
        self,
        owner: MaskPreviewWorker,
        image_rgb: np.ndarray,
        points: list[list[float]],
        labels: list[int],
        box: list[float] | None,
        refine: bool = False,
        model_id: str = "",
    ) -> None:
        super().__init__()
        self._owner = owner
        self._image = image_rgb
        self._points = points
        self._labels = labels
        self._box = box
        self._refine = refine
        self._model_id = model_id

    def run(self) -> None:
        try:
            mask = self._compute()
            self._owner._finish(mask, None)
        except Exception as e:
            self._owner._finish(None, f"{type(e).__name__}: {e}")

    def _compute(self) -> np.ndarray:
        owner = self._owner
        eng = owner._engine
        if not eng:
            owner.status.emit("Loading SAM 3 for preview (one-time)…")
            import torch
            from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
            eng["processor"] = Sam3TrackerVideoProcessor.from_pretrained("facebook/sam3")
            model: Any = Sam3TrackerVideoModel.from_pretrained("facebook/sam3", dtype=dtype)
            model.to(device).eval()
            eng["model"] = model
            eng["device"] = device
            eng["dtype"] = dtype

        owner.status.emit("Computing mask preview…")
        from PIL import Image

        processor = eng["processor"]
        model = eng["model"]

        h, w = int(self._image.shape[0]), int(self._image.shape[1])
        pil = Image.fromarray(
            (np.clip(self._image, 0.0, 1.0) * 255.0).astype(np.uint8), "RGB"
        )
        # Single-frame session — same call shapes as the batch pass.
        session = processor.init_video_session(
            video=[pil],
            inference_device=eng["device"],
            dtype=eng["dtype"],
        )
        prompt_kwargs: dict[str, Any] = {}
        if self._points:
            prompt_kwargs["input_points"] = [[self._points]]
            prompt_kwargs["input_labels"] = [[self._labels or [1] * len(self._points)]]
        if self._box is not None:
            prompt_kwargs["input_boxes"] = [[self._box]]
        processor.add_inputs_to_inference_session(
            session,
            frame_idx=0,
            obj_ids=1,
            original_size=(h, w),
            **prompt_kwargs,
        )
        out = np.zeros((h, w), dtype=np.float32)
        for step in model.propagate_in_video_iterator(session, start_frame_idx=0):
            pred_masks = getattr(step, "pred_masks", None)
            if pred_masks is None:
                continue
            pred_5d = pred_masks.unsqueeze(0) if pred_masks.ndim == 4 else pred_masks
            post = processor.post_process_masks(
                pred_5d, original_sizes=[(h, w)], mask_threshold=0.0, binarize=True
            )
            if not post:
                continue
            m = post[0]
            arr = m.float().cpu().numpy() if hasattr(m, "float") else np.asarray(m, np.float32)
            while arr.ndim > 2:
                arr = arr[0]
            out = arr.astype(np.float32, copy=False)

        # Optional soft-edge refinement — run BiRefNet on the seed frame so the
        # overlay previews the SOFT matte a full submit would produce, not the
        # hard SAM 3 mask. The refiner instance (and its resident model) is
        # cached in the engine so re-previews are fast; unload() frees it.
        if self._refine and float(out.sum()) > 0.0:
            owner.status.emit("Refining edges (BiRefNet)…")
            model_id = self._model_id or "ZhengPeng7/BiRefNet-portrait"
            refiner = eng.get("refiner")
            if refiner is None or eng.get("refiner_model_id") != model_id:
                from live_action_aov.passes.matte.birefnet import BiRefNetRefinerPass

                refiner = BiRefNetRefinerPass({"model_id": model_id})
                eng["refiner"] = refiner
                eng["refiner_model_id"] = model_id
            plate = self._image[None].astype(np.float32, copy=False)  # (1, H, W, 3)
            hard = out[None].astype(np.float32, copy=False)  # (1, H, W)
            soft = refiner._refine_instance(plate, hard)[0]
            out = np.clip(soft.astype(np.float32, copy=False), 0.0, 1.0)
        return out


__all__ = ["MaskPreviewWorker"]
