from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from detectors import create_face_detector
from pipeline.fas_predictor import FASPredictor
from pipeline.image_pipeline import process_image
from utils.io_utils import IMAGE_EXTENSIONS, config_base_dir, default_crop_dir, list_media_files, load_config, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Face Anti-Spoofing on an image or image folder.")
    parser.add_argument("--input", required=True, help="Input image path or folder.")
    parser.add_argument("--output", required=True, help="Output image path or output folder.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    return parser.parse_args()


def build_output_path(input_file: Path, output_arg: Path, input_is_dir: bool) -> Path:
    if input_is_dir:
        return output_arg / f"{input_file.stem}_result.jpg"
    if output_arg.suffix:
        return output_arg
    return output_arg / f"{input_file.stem}_result.jpg"


def collect_inputs(input_path: Path) -> List[Path]:
    if input_path.is_dir():
        files = list_media_files(input_path, IMAGE_EXTENSIONS)
        if not files:
            raise FileNotFoundError(f"No supported images found in folder: {input_path}")
        return files
    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")
    if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image extension: {input_path.suffix}")
    return [input_path]


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    input_path = Path(args.input)
    output_arg = Path(args.output)
    input_files = collect_inputs(input_path)
    output_arg.mkdir(parents=True, exist_ok=True) if input_path.is_dir() or not output_arg.suffix else output_arg.parent.mkdir(parents=True, exist_ok=True)

    predictor = FASPredictor(config)
    detector = create_face_detector(config)
    crop_dir = default_crop_dir(config, ui=False)

    try:
        for image_file in input_files:
            output_path = build_output_path(image_file, output_arg, input_is_dir=input_path.is_dir())
            result = process_image(
                input_path=image_file,
                output_path=output_path,
                config=config,
                predictor=predictor,
                detector=detector,
                crop_dir=crop_dir,
            )
            print(f"Saved result image: {result.output_path}")
            if result.json_path:
                print(f"Saved prediction JSON: {result.json_path}")
    finally:
        detector.close()

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
