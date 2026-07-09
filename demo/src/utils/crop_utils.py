from __future__ import annotations

from typing import Tuple

import numpy as np

BBox = Tuple[int, int, int, int]


def clip_bbox(bbox: BBox, image_width: int, image_height: int) -> BBox:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(round(x1)), image_width - 1))
    y1 = max(0, min(int(round(y1)), image_height - 1))
    x2 = max(0, min(int(round(x2)), image_width))
    y2 = max(0, min(int(round(y2)), image_height))
    return x1, y1, x2, y2


def is_valid_bbox(bbox: BBox) -> bool:
    x1, y1, x2, y2 = bbox
    return x2 > x1 and y2 > y1


def bbox_area(bbox: BBox) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def expand_bbox(bbox: BBox, margin: float, image_width: int, image_height: int) -> BBox:
    if margin < 1.0:
        raise ValueError(f"face_margin must be >= 1.0, got {margin}")

    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    center_x = x1 + width / 2.0
    center_y = y1 + height / 2.0
    new_width = width * margin
    new_height = height * margin

    expanded = (
        int(round(center_x - new_width / 2.0)),
        int(round(center_y - new_height / 2.0)),
        int(round(center_x + new_width / 2.0)),
        int(round(center_y + new_height / 2.0)),
    )
    return clip_bbox(expanded, image_width=image_width, image_height=image_height)


def crop_image(image_bgr: np.ndarray, bbox: BBox) -> np.ndarray:
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Cannot crop from an empty image.")
    if not is_valid_bbox(bbox):
        raise ValueError(f"Invalid crop bbox: {bbox}")

    x1, y1, x2, y2 = bbox
    crop = image_bgr[y1:y2, x1:x2].copy()
    if crop.size == 0:
        raise ValueError(f"Crop is empty for bbox: {bbox}")
    return crop
