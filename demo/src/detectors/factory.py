from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from detectors.base import FaceDetectorProtocol
from detectors.fallback_face_detector import FallbackFaceDetector
from detectors.mediapipe_face_detector import MediaPipeFaceDetector
from detectors.scrfd_face_detector import SCRFDFaceDetector


def _det_size(value: Any) -> tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    return 640, 640


def create_face_detector(config: Dict[str, Any]) -> FaceDetectorProtocol:
    detector_cfg = config.get("detector", {})
    detector_name = str(detector_cfg.get("name") or detector_cfg.get("type") or "scrfd").lower()
    min_confidence = float(detector_cfg.get("min_confidence") or detector_cfg.get("confidence_threshold") or 0.3)

    if detector_name == "scrfd":
        primary = SCRFDFaceDetector(
            min_confidence=min_confidence,
            model_name=detector_cfg.get("model_name", "buffalo_sc"),
            det_size=_det_size(detector_cfg.get("det_size", [640, 640])),
            root=detector_cfg.get("root", "~/.insightface"),
            providers=detector_cfg.get("providers"),
            ctx_id=detector_cfg.get("ctx_id"),
        )
        fallback_cfg = detector_cfg.get("fallback")
        use_fallback = bool(detector_cfg.get("fallback_to_mediapipe", True)) or (fallback_cfg is not None and str(fallback_cfg).lower() == "mediapipe")
        if use_fallback:
            fallback_confidence = float(detector_cfg.get("fallback_min_confidence", min_confidence))
            fallback = MediaPipeFaceDetector(min_confidence=fallback_confidence)
            return FallbackFaceDetector(primary=primary, fallback=fallback)
        return primary

    if detector_name == "mediapipe":
        return MediaPipeFaceDetector(min_confidence=min_confidence)

    raise ValueError(
        f"Unsupported detector name/type: {detector_name}. Supported values: scrfd, mediapipe."
    )