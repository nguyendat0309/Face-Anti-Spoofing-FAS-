from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Optional

import cv2
import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


def load_config(config_path: str | Path) -> Dict[str, Any]:
    path = Path(config_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    config["_config_path"] = str(path.resolve())
    config["_config_dir"] = str(path.resolve().parent)
    return config


def clone_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(config)


def deep_update(base: MutableMapping[str, Any], updates: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in updates.items():
        if isinstance(value, MutableMapping) and isinstance(base.get(key), MutableMapping):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def config_base_dir(config: Dict[str, Any]) -> Path:
    return Path(config.get("_config_dir", ".")).resolve()


def resolve_path(path_value: str | Path, base_dir: Optional[str | Path] = None) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    root = Path(base_dir).expanduser() if base_dir else Path.cwd()
    return (root / path).resolve()


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def read_image_bgr(path: str | Path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def write_image_bgr(path: str | Path, image_bgr) -> Path:
    path = ensure_parent(path)
    ok = cv2.imwrite(str(path), image_bgr)
    if not ok:
        raise IOError(f"Could not write image: {path}")
    return path


def save_json(path: str | Path, data: Dict[str, Any]) -> Path:
    path = ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def list_media_files(input_dir: str | Path, extensions: Iterable[str]) -> List[Path]:
    directory = Path(input_dir)
    exts = {ext.lower() for ext in extensions}
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in exts)


def safe_float(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def default_crop_dir(config: Dict[str, Any], ui: bool = False) -> Path:
    base = config_base_dir(config)
    if ui:
        rel = Path("outputs/ui/crops")
    else:
        configured = config.get("output", {}).get("crop_dir") or config.get("crop", {}).get("crop_dir")
        rel = Path(configured) if configured else Path("outputs/crops")
    return resolve_path(rel, base)


def output_enabled(config: Dict[str, Any], key: str, default: bool = True) -> bool:
    return bool(config.get("output", {}).get(key, default))
