from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from detectors import create_face_detector
from pipeline.fas_predictor import FASPredictor
from pipeline.video_pipeline import process_video
from utils.io_utils import VIDEO_EXTENSIONS, default_crop_dir, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Face Anti-Spoofing on a video.")
    parser.add_argument("--input", required=True, help="Input video path.")
    parser.add_argument("--output", required=True, help="Output video path or output folder.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    return parser.parse_args()


def build_output_path(input_file: Path, output_arg: Path) -> Path:
    if output_arg.suffix:
        return output_arg
    return output_arg / f"{input_file.stem}_result.mp4"


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    if input_path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported video extension: {input_path.suffix}")

    output_path = build_output_path(input_path, Path(args.output))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    predictor = FASPredictor(config)
    detector = create_face_detector(config)
    crop_dir = default_crop_dir(config, ui=False)

    try:
        result = process_video(
            input_path=input_path,
            output_path=output_path,
            config=config,
            predictor=predictor,
            detector=detector,
            crop_dir=crop_dir,
            show_progress=True,
        )
    finally:
        detector.close()

    print("Video processed successfully.")
    print(f"Total frames: {result.total_frames}")
    print(f"Frames with face: {result.frames_with_face}")
    print(f"Frames without face: {result.frames_without_face}")
    print(f"Output video saved to: {result.output_path}")
    if result.csv_path:
        print(f"Prediction CSV saved to: {result.csv_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
