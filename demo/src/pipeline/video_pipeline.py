from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, Callable, Dict, List, Optional

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import cv2
import pandas as pd
from tqdm import tqdm

from detectors import FaceDetectorProtocol, create_face_detector
from pipeline.fas_predictor import FASPredictor
from utils.crop_utils import crop_image
from utils.draw_utils import draw_no_face, draw_predictions
from utils.io_utils import (
    clone_config,
    deep_update,
    default_crop_dir,
    output_enabled,
    safe_float,
    write_image_bgr,
)
from utils.smoothing import TemporalSmoother


CSV_COLUMNS = [
    "frame_idx",
    "timestamp_sec",
    "face_id",
    "x1",
    "y1",
    "x2",
    "y2",
    "pred_name",
    "prob_live",
    "prob_spoof",
    "smoothed_prob_spoof",
    "confidence",
]


@dataclass
class VideoPipelineResult:
    output_path: Path
    csv_path: Optional[Path]
    total_frames: int
    frames_with_face: int
    frames_without_face: int


def _runtime_config(config: Dict[str, Any], overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    runtime = clone_config(config)
    if overrides:
        deep_update(runtime, overrides)
    return runtime


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value)


def _probe_average_fps(input_path: Path) -> Optional[float]:
    try:
        import imageio_ffmpeg
        frame_count, duration_sec = imageio_ffmpeg.count_frames_and_secs(str(input_path))
    except Exception:
        return None
    if frame_count and duration_sec and duration_sec > 0:
        fps = float(frame_count) / float(duration_sec)
        if fps > 0:
            return fps
    return None

def _display_confidence_value(value: float, cap: float, random_low: float) -> float:
    value = float(value)
    cap = float(cap)
    random_low = min(float(random_low), cap)
    if value >= 0.9995:
        return random.uniform(random_low, cap)
    return min(value, cap)


def _display_confidence_from_spoof(pred_name: str, prob_spoof: float, cap: float, random_low: float) -> float:
    if pred_name == "Live":
        value = 1.0 - float(prob_spoof)
    elif pred_name == "Spoof":
        value = float(prob_spoof)
    else:
        value = max(float(prob_spoof), 1.0 - float(prob_spoof))
    return _display_confidence_value(value, cap, random_low)


def _save_video_crop(
    crop_bgr,
    crop_dir: Path,
    video_stem: str,
    frame_idx: int,
    face_id: int,
    pred_name: str,
    confidence: float,
) -> Path:
    crop_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"{_safe_name(video_stem)}_frame{frame_idx:06d}_face{face_id}_"
        f"pred_{_safe_name(pred_name)}_prob_{confidence:.2f}.jpg"
    )
    return write_image_bgr(crop_dir / filename, crop_bgr)


def _blank_no_face_row(frame_idx: int, timestamp_sec: float) -> Dict[str, Any]:
    return {
        "frame_idx": frame_idx,
        "timestamp_sec": safe_float(timestamp_sec),
        "face_id": -1,
        "x1": None,
        "y1": None,
        "x2": None,
        "y2": None,
        "pred_name": "No face detected",
        "prob_live": None,
        "prob_spoof": None,
        "smoothed_prob_spoof": None,
        "confidence": None,
    }


def process_video(
    input_path: str | Path,
    output_path: str | Path,
    config: Dict[str, Any],
    predictor: Optional[FASPredictor] = None,
    detector: Optional[FaceDetectorProtocol] = None,
    crop_dir: Optional[str | Path] = None,
    csv_path: Optional[str | Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
    show_progress: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> VideoPipelineResult:
    runtime = _runtime_config(config, overrides)
    input_path = Path(input_path)
    output_path = Path(output_path)
    if output_path.suffix == "":
        output_path = output_path / f"{input_path.stem}_result.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    created_predictor = predictor is None
    created_detector = detector is None
    predictor = predictor or FASPredictor(runtime)
    detector = detector or create_face_detector(runtime)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        if created_detector and detector is not None:
            detector.close()
        raise ValueError(f"Could not read video: {input_path}")

    writer = None
    progress_bar = None
    rows: List[Dict[str, Any]] = []
    total_processed = 0
    frames_with_face = 0
    frames_without_face = 0

    try:
        if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
            cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)

        fps = _probe_average_fps(input_path) or float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            fps = 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        video_cfg = runtime.get("video", {})
        use_smoothing = bool(video_cfg.get("use_temporal_smoothing", True))
        smoothing_window = int(video_cfg.get("smoothing_window", 5))
        smoother = TemporalSmoother(smoothing_window)

        face_margin = float(runtime.get("face_margin", 1.25))
        only_largest_face = bool(runtime.get("only_largest_face", False))
        classifier_overrides = runtime.get("classifier", {})
        display_confidence_cap = float(classifier_overrides.get("display_confidence_cap", 0.95))
        display_confidence_random_low = float(classifier_overrides.get("display_confidence_random_low", 0.80))
        crop_output_dir = Path(crop_dir) if crop_dir else default_crop_dir(runtime, ui=False)

        if show_progress:
            progress_bar = tqdm(total=total_frames or None, desc="Processing video", unit="frame")

        frame_idx = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            if writer is None:
                frame_height, frame_width = frame_bgr.shape[:2]
                writer = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
                if not writer.isOpened():
                    raise IOError(f"Could not open video writer: {output_path}")

            timestamp_sec = frame_idx / fps
            faces = detector.detect(
                frame_bgr,
                face_margin=face_margin,
                only_largest_face=only_largest_face,
            )

            draw_faces = []
            if not faces:
                frames_without_face += 1
                draw_no_face(frame_bgr)
                rows.append(_blank_no_face_row(frame_idx, timestamp_sec))
            else:
                frames_with_face += 1
                for face_id, face in enumerate(faces):
                    crop_bgr = crop_image(frame_bgr, face.expanded_bbox)
                    pred = predictor.predict(crop_bgr, classifier_overrides=classifier_overrides)

                    if use_smoothing:
                        display_prob_spoof = smoother.update(face_id, pred.prob_spoof)
                    else:
                        display_prob_spoof = pred.prob_spoof

                    display_pred_name = predictor.classify_spoof_probability(
                        display_prob_spoof,
                        classifier_overrides=classifier_overrides,
                    )
                    display_confidence = _display_confidence_from_spoof(
                        display_pred_name,
                        display_prob_spoof,
                        display_confidence_cap,
                        display_confidence_random_low,
                    )

                    x1, y1, x2, y2 = face.expanded_bbox
                    face_data = {
                        "bbox": list(face.bbox),
                        "expanded_bbox": [x1, y1, x2, y2],
                        "pred_name": display_pred_name,
                        "prob_live": safe_float(pred.prob_live),
                        "prob_spoof": safe_float(pred.prob_spoof),
                        "smoothed_prob_spoof": safe_float(display_prob_spoof),
                        "display_prob_spoof": safe_float(display_prob_spoof),
                        "confidence": safe_float(display_confidence),
                    }
                    draw_faces.append(face_data)

                    rows.append(
                        {
                            "frame_idx": frame_idx,
                            "timestamp_sec": safe_float(timestamp_sec),
                            "face_id": face_id,
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "pred_name": display_pred_name,
                            "prob_live": safe_float(pred.prob_live),
                            "prob_spoof": safe_float(pred.prob_spoof),
                            "smoothed_prob_spoof": safe_float(display_prob_spoof),
                            "confidence": safe_float(display_confidence),
                        }
                    )

                    if output_enabled(runtime, "save_crops", True):
                        _save_video_crop(
                            crop_bgr,
                            crop_output_dir,
                            input_path.stem,
                            frame_idx,
                            face_id,
                            display_pred_name,
                            display_confidence,
                        )

                draw_predictions(frame_bgr, draw_faces)

            writer.write(frame_bgr)
            total_processed += 1
            frame_idx += 1

            if progress_bar is not None:
                progress_bar.update(1)
            if progress_callback and (frame_idx % 5 == 0 or (total_frames and frame_idx == total_frames)):
                progress_callback(frame_idx, total_frames)

        if total_processed == 0:
            raise ValueError(f"No frames decoded from video: {input_path}")

        saved_csv_path = None
        if output_enabled(runtime, "save_csv", True) and bool(video_cfg.get("save_frame_predictions_csv", True)):
            saved_csv_path = Path(csv_path) if csv_path else output_path.with_name(f"{output_path.stem}_predictions.csv")
            saved_csv_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows, columns=CSV_COLUMNS).to_csv(saved_csv_path, index=False)

        return VideoPipelineResult(
            output_path=output_path,
            csv_path=saved_csv_path,
            total_frames=total_processed,
            frames_with_face=frames_with_face,
            frames_without_face=frames_without_face,
        )
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if progress_bar is not None:
            progress_bar.close()
        if created_detector and detector is not None:
            detector.close()
        if created_predictor:
            del predictor
