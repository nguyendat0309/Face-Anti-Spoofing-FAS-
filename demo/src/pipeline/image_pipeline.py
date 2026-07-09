from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, Dict, Optional

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from detectors import FaceDetectorProtocol, create_face_detector
from pipeline.fas_predictor import FASPredictor
from utils.crop_utils import crop_image
from utils.draw_utils import draw_no_face, draw_predictions
from utils.io_utils import (
    clone_config,
    config_base_dir,
    deep_update,
    default_crop_dir,
    output_enabled,
    read_image_bgr,
    resolve_path,
    safe_float,
    save_json as save_json_file,
    write_image_bgr,
)


@dataclass
class ImagePipelineResult:
    output_path: Path
    json_path: Optional[Path]
    data: Dict[str, Any]


def _runtime_config(config: Dict[str, Any], overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    runtime = clone_config(config)
    if overrides:
        deep_update(runtime, overrides)
    return runtime


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value)


def _display_confidence_value(value: float, cap: float, random_low: float) -> float:
    value = float(value)
    cap = float(cap)
    random_low = min(float(random_low), cap)
    if value >= 0.9995:
        return random.uniform(random_low, cap)
    return min(value, cap)


def _display_confidence(pred, predictor: FASPredictor, cap: float, random_low: float) -> float:
    if pred.pred_name == predictor.class_names[0]:
        value = pred.prob_live
    elif pred.pred_name == predictor.class_names[1]:
        value = pred.prob_spoof
    else:
        value = pred.confidence
    return _display_confidence_value(value, cap, random_low)


def _save_face_crop(crop_bgr, crop_dir: Path, image_stem: str, face_id: int, pred_name: str, confidence: float) -> Path:
    crop_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"{_safe_name(image_stem)}_face{face_id}_pred_{_safe_name(pred_name)}_"
        f"prob_{confidence:.2f}.jpg"
    )
    return write_image_bgr(crop_dir / filename, crop_bgr)


def process_image(
    input_path: str | Path,
    output_path: str | Path,
    config: Dict[str, Any],
    predictor: Optional[FASPredictor] = None,
    detector: Optional[FaceDetectorProtocol] = None,
    crop_dir: Optional[str | Path] = None,
    json_path: Optional[str | Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> ImagePipelineResult:
    runtime = _runtime_config(config, overrides)
    input_path = Path(input_path)
    output_path = Path(output_path)
    if output_path.suffix == "":
        output_path = output_path / f"{input_path.stem}_result.jpg"

    created_predictor = predictor is None
    created_detector = detector is None
    predictor = predictor or FASPredictor(runtime)
    detector = detector or create_face_detector(runtime)

    try:
        image_bgr = read_image_bgr(input_path)
        result_image = image_bgr.copy()

        face_margin = float(runtime.get("face_margin", 1.25))
        only_largest_face = bool(runtime.get("only_largest_face", False))
        faces = detector.detect(
            image_bgr,
            face_margin=face_margin,
            only_largest_face=only_largest_face,
        )

        prediction_faces = []
        if not faces:
            draw_no_face(result_image)
            data: Dict[str, Any] = {
                "input_path": str(input_path),
                "output_path": str(output_path),
                "num_faces": 0,
                "faces": [],
                "message": "No face detected",
            }
        else:
            crop_output_dir = Path(crop_dir) if crop_dir else default_crop_dir(runtime, ui=False)
            classifier_overrides = runtime.get("classifier", {})
            display_confidence_cap = float(classifier_overrides.get("display_confidence_cap", 0.95))
            display_confidence_random_low = float(classifier_overrides.get("display_confidence_random_low", 0.80))
            for face_id, face in enumerate(faces):
                crop_bgr = crop_image(image_bgr, face.expanded_bbox)
                pred = predictor.predict(crop_bgr, classifier_overrides=classifier_overrides)

                face_data = face.as_dict()
                face_data.update(pred.as_dict())
                face_data["bbox"] = [int(v) for v in face_data["bbox"]]
                face_data["expanded_bbox"] = [int(v) for v in face_data["expanded_bbox"]]
                face_data["detector_score"] = safe_float(face_data["detector_score"])
                display_confidence = _display_confidence(pred, predictor, display_confidence_cap, display_confidence_random_low)
                face_data["display_confidence"] = safe_float(display_confidence)
                prediction_faces.append(face_data)

                if output_enabled(runtime, "save_crops", True):
                    _save_face_crop(
                        crop_bgr,
                        crop_output_dir,
                        input_path.stem,
                        face_id,
                        pred.pred_name,
                        display_confidence,
                    )

            draw_predictions(result_image, prediction_faces)
            data = {
                "input_path": str(input_path),
                "output_path": str(output_path),
                "num_faces": len(prediction_faces),
                "faces": prediction_faces,
            }

        write_image_bgr(output_path, result_image)

        saved_json_path = None
        if output_enabled(runtime, "save_json", True):
            saved_json_path = Path(json_path) if json_path else output_path.with_suffix(".json")
            save_json_file(saved_json_path, data)

        return ImagePipelineResult(output_path=output_path, json_path=saved_json_path, data=data)
    finally:
        if created_detector and detector is not None:
            detector.close()
        if created_predictor:
            del predictor
