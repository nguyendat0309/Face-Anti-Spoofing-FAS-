from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

from detectors.base import FaceBox
from utils.crop_utils import clip_bbox, expand_bbox, is_valid_bbox


class SCRFDFaceDetector:
    """SCRFD face detector through InsightFace FaceAnalysis.

    The default `buffalo_sc` pack uses a small SCRFD detector (`det_500m.onnx`),
    which is a good local-demo tradeoff between crop quality and speed.
    """

    def __init__(
        self,
        min_confidence: float = 0.5,
        model_name: str = "buffalo_sc",
        det_size: tuple[int, int] = (640, 640),
        root: str = "~/.insightface",
        providers: Optional[Iterable[str]] = None,
        ctx_id: Optional[int] = None,
    ):
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise ImportError(
                "insightface is required for SCRFD face detection. Install dependencies with: "
                "pip install -r requirements.txt"
            ) from exc

        self.min_confidence = float(min_confidence)
        self.model_name = str(model_name)
        self.det_size = tuple(int(v) for v in det_size)
        self.root = str(Path(root).expanduser())
        self.providers = self._select_providers(providers)
        self.ctx_id = self._select_ctx_id(ctx_id)

        self._app = FaceAnalysis(
            name=self.model_name,
            root=self.root,
            allowed_modules=["detection"],
            providers=self.providers,
        )
        self._app.prepare(
            ctx_id=self.ctx_id,
            det_thresh=self.min_confidence,
            det_size=self.det_size,
        )

    @staticmethod
    def _select_providers(providers: Optional[Iterable[str]]) -> list[str]:
        requested = list(providers or [])
        try:
            import onnxruntime as ort
            available = set(ort.get_available_providers())
        except Exception:
            available = set()

        if requested:
            selected = [provider for provider in requested if not available or provider in available]
            if selected:
                return selected

        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def _select_ctx_id(self, ctx_id: Optional[int]) -> int:
        if ctx_id is not None:
            return int(ctx_id)
        return 0 if any(provider == "CUDAExecutionProvider" for provider in self.providers) else -1

    def detect(
        self,
        image_bgr: np.ndarray,
        face_margin: float = 1.25,
        only_largest_face: bool = False,
    ) -> List[FaceBox]:
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Cannot run face detection on an empty image/frame.")

        image_height, image_width = image_bgr.shape[:2]
        detections = self._app.get(image_bgr)
        faces: List[FaceBox] = []

        for detection in detections:
            score = float(getattr(detection, "det_score", 0.0))
            if score < self.min_confidence:
                continue

            x1, y1, x2, y2 = getattr(detection, "bbox")
            bbox = clip_bbox((x1, y1, x2, y2), image_width=image_width, image_height=image_height)
            if not is_valid_bbox(bbox):
                continue

            expanded = expand_bbox(
                bbox,
                margin=face_margin,
                image_width=image_width,
                image_height=image_height,
            )
            if not is_valid_bbox(expanded):
                continue

            faces.append(FaceBox(bbox=bbox, expanded_bbox=expanded, score=score))

        if only_largest_face and faces:
            faces = [max(faces, key=lambda face: face.area)]

        return faces

    def close(self) -> None:
        return None
