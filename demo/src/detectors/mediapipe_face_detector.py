from __future__ import annotations

from typing import List

import cv2
import numpy as np

from detectors.base import FaceBox
from utils.crop_utils import clip_bbox, expand_bbox, is_valid_bbox


class MediaPipeFaceDetector:
    def __init__(self, min_confidence: float = 0.5):
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise ImportError(
                "mediapipe is required for face detection. Install dependencies with: "
                "pip install -r requirements.txt"
            ) from exc

        self.min_confidence = float(min_confidence)
        self._detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=self.min_confidence,
        )

    def detect(
        self,
        image_bgr: np.ndarray,
        face_margin: float = 1.25,
        only_largest_face: bool = False,
    ) -> List[FaceBox]:
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Cannot run face detection on an empty image/frame.")

        image_height, image_width = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = self._detector.process(image_rgb)
        faces: List[FaceBox] = []

        if not result.detections:
            return faces

        for detection in result.detections:
            relative_bbox = detection.location_data.relative_bounding_box
            x1 = int(round(relative_bbox.xmin * image_width))
            y1 = int(round(relative_bbox.ymin * image_height))
            x2 = int(round((relative_bbox.xmin + relative_bbox.width) * image_width))
            y2 = int(round((relative_bbox.ymin + relative_bbox.height) * image_height))
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

            score = float(detection.score[0]) if detection.score else 0.0
            faces.append(FaceBox(bbox=bbox, expanded_bbox=expanded, score=score))

        if only_largest_face and faces:
            faces = [max(faces, key=lambda face: face.area)]

        return faces

    def close(self) -> None:
        self._detector.close()
