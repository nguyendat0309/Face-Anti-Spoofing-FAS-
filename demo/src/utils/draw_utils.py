from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import cv2
import numpy as np

Color = Tuple[int, int, int]

LABEL_COLORS: Dict[str, Color] = {
    "Live": (60, 180, 75),
    "Spoof": (40, 40, 230),
    "Uncertain": (0, 180, 255),
    "No face detected": (150, 150, 150),
}


def color_for_label(label: str) -> Color:
    return LABEL_COLORS.get(label, (255, 255, 255))


def format_label(pred_name: str, confidence: float) -> str:
    return f"{pred_name} | conf={confidence:.2f}"


def draw_text_box(
    image_bgr: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    color: Color,
    font_scale: float = 0.6,
    thickness: int = 2,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = origin
    text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
    text_width, text_height = text_size

    y_top = max(0, y - text_height - baseline - 6)
    y_bottom = max(text_height + baseline + 6, y)
    if y_top == 0:
        y_bottom = text_height + baseline + 8

    cv2.rectangle(
        image_bgr,
        (x, y_top),
        (min(image_bgr.shape[1] - 1, x + text_width + 8), y_bottom),
        color,
        thickness=-1,
    )
    cv2.putText(
        image_bgr,
        text,
        (x + 4, y_bottom - baseline - 4),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        lineType=cv2.LINE_AA,
    )


def draw_predictions(image_bgr: np.ndarray, faces: Iterable[Dict[str, Any]]) -> np.ndarray:
    for face in faces:
        pred_name = str(face["pred_name"])
        confidence = float(face.get("display_confidence", face.get("confidence", 0.0)))
        x1, y1, x2, y2 = face.get("expanded_bbox", face["bbox"])
        color = color_for_label(pred_name)

        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, thickness=2)
        draw_text_box(image_bgr, format_label(pred_name, confidence), (x1, y1), color)
    return image_bgr


def draw_no_face(image_bgr: np.ndarray) -> np.ndarray:
    text = "No face detected"
    color = color_for_label(text)
    draw_text_box(image_bgr, text, (20, 42), color)
    return image_bgr
