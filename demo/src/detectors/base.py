from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from utils.crop_utils import BBox, bbox_area


@dataclass
class FaceBox:
    bbox: BBox
    expanded_bbox: BBox
    score: float

    @property
    def area(self) -> int:
        return bbox_area(self.bbox)

    def as_dict(self) -> dict:
        return {
            "bbox": list(self.bbox),
            "expanded_bbox": list(self.expanded_bbox),
            "detector_score": float(self.score),
        }


class FaceDetectorProtocol(Protocol):
    def detect(
        self,
        image_bgr: np.ndarray,
        face_margin: float = 1.25,
        only_largest_face: bool = False,
    ) -> list[FaceBox]:
        ...

    def close(self) -> None:
        ...
