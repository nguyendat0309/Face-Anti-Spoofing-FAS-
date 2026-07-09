from __future__ import annotations

import argparse
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from detectors import create_face_detector
from pipeline.fas_predictor import FASPredictor
from pipeline.image_pipeline import process_image
from pipeline.video_pipeline import process_video
from utils.io_utils import config_base_dir, ensure_dir, load_config, resolve_path


DESCRIPTION = (
    "Upload an image or video. The system detects faces, crops them, classifies each face as "
    "Live, Spoof, or Uncertain, and visualizes the result."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Gradio Face Anti-Spoofing demo.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    return parser.parse_args()


def find_available_port(host: str, preferred_port: int, max_tries: int = 50) -> int:
    for port in range(preferred_port, preferred_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return port
    raise OSError(
        f"No available Gradio port found in range {preferred_port}-{preferred_port + max_tries - 1}."
    )


def uploaded_path(value: Any) -> Optional[Path]:
    if value is None:
        return None
    if isinstance(value, str):
        return Path(value)
    if isinstance(value, dict):
        path_value = value.get("path") or value.get("name")
        return Path(path_value) if path_value else None
    path_value = getattr(value, "path", None) or getattr(value, "name", None)
    if path_value:
        return Path(path_value)
    if isinstance(value, (list, tuple)) and value:
        return uploaded_path(value[0])
    return None


def make_browser_playable_video(video_path: Path) -> Path:
    try:
        import imageio_ffmpeg
    except ImportError:
        return video_path

    output_path = video_path.with_name(f"{video_path.stem}_browser.mp4")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        return video_path
    return output_path if output_path.exists() else video_path


def build_overrides(
    only_largest_face: bool,
    face_margin: float,
    spoof_threshold: float,
    live_threshold: float,
    use_temporal_smoothing: Optional[bool] = None,
    smoothing_window: Optional[int] = None,
) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {
        "only_largest_face": bool(only_largest_face),
        "face_margin": float(face_margin),
        "classifier": {
            "spoof_threshold": float(spoof_threshold),
            "live_threshold": float(live_threshold),
        },
    }
    if use_temporal_smoothing is not None:
        overrides["video"] = {
            "use_temporal_smoothing": bool(use_temporal_smoothing),
            "smoothing_window": int(smoothing_window or 1),
        }
    return overrides


def create_app(config: Dict[str, Any]) -> Any:
    try:
        import gradio as gr
    except ImportError as exc:
        raise ImportError(
            "gradio is required for the UI. Install dependencies with: pip install -r requirements.txt"
        ) from exc

    predictor = FASPredictor(config)
    detector = create_face_detector(config)

    base_dir = config_base_dir(config)
    image_output_dir = ensure_dir(resolve_path("outputs/ui/images", base_dir))
    video_output_dir = ensure_dir(resolve_path("outputs/ui/videos", base_dir))
    crop_output_dir = ensure_dir(resolve_path("outputs/ui/crops", base_dir))

    classifier_cfg = config.get("classifier", {})
    prediction_cfg = config.get("prediction", {})
    video_cfg = config.get("video", {})
    ui_cfg = config.get("ui", {})
    title = ui_cfg.get("title", "Face Anti-Spoofing Demo")

    default_margin = float(config.get("face_margin") or config.get("crop", {}).get("margin", 1.35))
    default_spoof_thresh = float(prediction_cfg.get("spoof_threshold") or classifier_cfg.get("spoof_threshold", 0.65))
    default_live_thresh = float(prediction_cfg.get("live_threshold") or classifier_cfg.get("live_threshold", 0.35))
    default_smoothing_window = int(video_cfg.get("smoothing_window", 5))

    def run_image_demo(
        image_input,
        only_largest_face,
        face_margin,
        spoof_threshold,
        live_threshold,
    ):
        input_path = uploaded_path(image_input)
        if input_path is None:
            return None

        output_path = image_output_dir / f"{input_path.stem}_result.jpg"
        overrides = build_overrides(
            only_largest_face=only_largest_face,
            face_margin=face_margin,
            spoof_threshold=spoof_threshold,
            live_threshold=live_threshold,
        )
        result = process_image(
            input_path=input_path,
            output_path=output_path,
            config=config,
            predictor=predictor,
            detector=detector,
            crop_dir=crop_output_dir,
            overrides=overrides,
        )
        return str(result.output_path)

    def run_video_demo(
        video_input,
        only_largest_face,
        use_temporal_smoothing,
        smoothing_window,
        face_margin,
        spoof_threshold,
        live_threshold,
        progress=gr.Progress(track_tqdm=False),
    ):
        input_path = uploaded_path(video_input)
        if input_path is None:
            return None

        output_path = video_output_dir / f"{input_path.stem}_result.mp4"
        overrides = build_overrides(
            only_largest_face=only_largest_face,
            face_margin=face_margin,
            spoof_threshold=spoof_threshold,
            live_threshold=live_threshold,
            use_temporal_smoothing=use_temporal_smoothing,
            smoothing_window=smoothing_window,
        )
        overrides["output"] = {"save_csv": False}

        progress(0.02, desc="Preparing video")

        def on_progress(done: int, total: int) -> None:
            if total > 0:
                progress(min(0.98, done / total), desc=f"Processing frame {done}/{total}")
            else:
                progress(0.5, desc=f"Processing frame {done}")

        result = process_video(
            input_path=input_path,
            output_path=output_path,
            config=config,
            predictor=predictor,
            detector=detector,
            crop_dir=crop_output_dir,
            overrides=overrides,
            show_progress=False,
            progress_callback=on_progress,
        )
        progress(1.0, desc="Done")
        return str(make_browser_playable_video(result.output_path))

    with gr.Blocks(title=title) as demo:
        gr.Markdown(f"# {title}")
        gr.Markdown(DESCRIPTION)

        with gr.Tab("Image Demo"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(label="Upload image", type="filepath")
                    with gr.Accordion("Settings", open=False):
                        image_only_largest = gr.Checkbox(
                            label="Only largest face",
                            value=bool(config.get("only_largest_face", False)),
                        )
                        image_margin = gr.Slider(
                            label="Face margin",
                            minimum=1.0,
                            maximum=3.0,
                            step=0.05,
                            value=default_margin,
                        )
                        image_spoof_threshold = gr.Slider(
                            label="Spoof threshold",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.01,
                            value=default_spoof_thresh,
                        )
                        image_live_threshold = gr.Slider(
                            label="Live threshold",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.01,
                            value=default_live_thresh,
                        )
                    image_button = gr.Button("Run Image Prediction", variant="primary")
                with gr.Column():
                    image_output = gr.Image(label="Image result", type="filepath")

            image_button.click(
                fn=run_image_demo,
                inputs=[
                    image_input,
                    image_only_largest,
                    image_margin,
                    image_spoof_threshold,
                    image_live_threshold,
                ],
                outputs=[image_output],
            )

        with gr.Tab("Video Demo"):
            with gr.Row():
                with gr.Column():
                    video_input = gr.Video(label="Upload video")
                    with gr.Accordion("Settings", open=False):
                        video_only_largest = gr.Checkbox(
                            label="Only largest face",
                            value=bool(config.get("only_largest_face", False)),
                        )
                        video_smoothing = gr.Checkbox(
                            label="Use temporal smoothing",
                            value=bool(video_cfg.get("use_temporal_smoothing", True)),
                        )
                        video_smoothing_window = gr.Slider(
                            label="Smoothing window",
                            minimum=1,
                            maximum=30,
                            step=1,
                            value=default_smoothing_window,
                        )
                        video_margin = gr.Slider(
                            label="Face margin",
                            minimum=1.0,
                            maximum=3.0,
                            step=0.05,
                            value=default_margin,
                        )
                        video_spoof_threshold = gr.Slider(
                            label="Spoof threshold",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.01,
                            value=default_spoof_thresh,
                        )
                        video_live_threshold = gr.Slider(
                            label="Live threshold",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.01,
                            value=default_live_thresh,
                        )
                    video_button = gr.Button("Run Video Prediction", variant="primary")
                with gr.Column():
                    video_output = gr.Video(label="Video result", format="mp4")

            video_button.click(
                fn=run_video_demo,
                inputs=[
                    video_input,
                    video_only_largest,
                    video_smoothing,
                    video_smoothing_window,
                    video_margin,
                    video_spoof_threshold,
                    video_live_threshold,
                ],
                outputs=[video_output],
            )

    return demo


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    app = create_app(config)
    ui_cfg = config.get("ui", {})
    server_name = ui_cfg.get("server_name", "127.0.0.1")
    preferred_port = int(ui_cfg.get("server_port") or 7860)
    server_port = find_available_port(server_name, preferred_port)
    if server_port != preferred_port:
        print(f"Configured Gradio port {preferred_port} is busy. Using port {server_port} instead.")
    app.launch(
        server_name=server_name,
        server_port=server_port,
        share=bool(ui_cfg.get("share", False)),
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        raise SystemExit(f"ERROR: {exc}") from exc