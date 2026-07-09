from __future__ import annotations

import sys
from pathlib import Path
from typing import List

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np

from detectors.base import FaceBox, FaceDetectorProtocol


class FallbackFaceDetector:
    """Use a secondary detector only when the primary detector finds no face."""

    def __init__(self, primary: FaceDetectorProtocol, fallback: FaceDetectorProtocol):
        self.primary = primary
        self.fallback = fallback

    def detect(
        self,
        image_bgr: np.ndarray,
        face_margin: float = 1.25,
        only_largest_face: bool = False,
    ) -> List[FaceBox]:
        faces = self.primary.detect(
            image_bgr,
            face_margin=face_margin,
            only_largest_face=only_largest_face,
        )
        if faces:
            return faces
        return self.fallback.detect(
            image_bgr,
            face_margin=face_margin,
            only_largest_face=only_largest_face,
        )

    def close(self) -> None:
        self.primary.close()
        self.fallback.close()
