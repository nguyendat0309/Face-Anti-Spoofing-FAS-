from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import cv2
import torch
from PIL import Image
from torchvision import transforms

from models.resnet18_gem_cbam import create_model
from utils.io_utils import config_base_dir, resolve_path, safe_float


@dataclass
class FASPrediction:
    pred_name: str
    prob_live: float
    prob_spoof: float
    confidence: float

    def as_dict(self) -> Dict[str, float | str]:
        return {
            "pred_name": self.pred_name,
            "prob_live": safe_float(self.prob_live),
            "prob_spoof": safe_float(self.prob_spoof),
            "confidence": safe_float(self.confidence),
        }


class FASPredictor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        classifier_cfg = config.get("classifier", {})
        self.class_names = classifier_cfg.get("class_names", ["Live", "Spoof"])
        if len(self.class_names) != 2:
            raise ValueError("classifier.class_names must contain exactly two classes: Live and Spoof.")

        model_cfg = config.get("model", {})
        device_val = config.get("device") or model_cfg.get("device", "auto")
        self.device = self._select_device(device_val)
        
        img_size_val = config.get("image_size") or model_cfg.get("image_size", 224)
        self.image_size = int(img_size_val)
        
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        num_classes = int(model_cfg.get("num_classes", 2))
        self.model = create_model(num_classes=num_classes).to(self.device)
        
        ckpt_path = config.get("checkpoint_path") or model_cfg.get("checkpoint_path", "checkpoints/resnet18_best.pth")
        self._load_checkpoint(ckpt_path)
        self.model.eval()

    @staticmethod
    def _select_device(device_value: str) -> torch.device:
        requested = str(device_value or "cpu").lower()
        if requested.startswith("cuda") or requested == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        return torch.device(requested)

    def _resolve_checkpoint_path(self, checkpoint_path: str | Path) -> Path:
        if checkpoint_path is None or str(checkpoint_path).strip() == "":
            raise ValueError(
                "checkpoint_path is empty. Set checkpoint_path in config.yaml to your fine-tuned .pth file."
            )

        resolved = resolve_path(checkpoint_path, base_dir=config_base_dir(self.config))
        if not resolved.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {resolved}")
        if not resolved.is_file():
            raise ValueError(f"checkpoint_path must point to a file, got: {resolved}")
        return resolved

    def _load_checkpoint(self, checkpoint_path: str | Path) -> None:
        resolved = self._resolve_checkpoint_path(checkpoint_path)
        try:
            checkpoint = torch.load(resolved, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(resolved, map_location=self.device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        filtered_state_dict = {}
        for k, v in state_dict.items():
            if (
                k.startswith("projection_head")
                or k.startswith("supcon_head")
                or k.startswith("module.projection_head")
                or k.startswith("module.supcon_head")
            ):
                continue
            new_key = k.replace("module.", "")
            filtered_state_dict[new_key] = v

        try:
            self.model.load_state_dict(filtered_state_dict, strict=False)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Could not load checkpoint into ResNet18GeMCBAM from {resolved}. Details: {exc}"
            ) from exc

    def classify_spoof_probability(
        self,
        prob_spoof: float,
        classifier_overrides: Optional[Dict[str, Any]] = None,
    ) -> str:
        classifier_cfg = dict(self.config.get("classifier", {}))
        prediction_cfg = dict(self.config.get("prediction", {}))
        if classifier_overrides:
            classifier_cfg.update(classifier_overrides)
            prediction_cfg.update(classifier_overrides)

        use_uncertain = prediction_cfg.get("uncertain_enabled", classifier_cfg.get("use_uncertain", True))
        use_uncertain = bool(use_uncertain)
        if not use_uncertain:
            return self.class_names[1] if prob_spoof >= 0.5 else self.class_names[0]

        live_threshold = float(prediction_cfg.get("live_threshold") or classifier_cfg.get("live_threshold", 0.35))
        spoof_threshold = float(prediction_cfg.get("spoof_threshold") or classifier_cfg.get("spoof_threshold", 0.65))
        if live_threshold > spoof_threshold:
            raise ValueError(
                f"live_threshold must be <= spoof_threshold, got {live_threshold} and {spoof_threshold}"
            )

        if prob_spoof >= spoof_threshold:
            return self.class_names[1]
        if prob_spoof <= live_threshold:
            return self.class_names[0]
        return "Uncertain"

    def predict(
        self,
        face_crop_bgr,
        classifier_overrides: Optional[Dict[str, Any]] = None,
    ) -> FASPrediction:
        if face_crop_bgr is None or face_crop_bgr.size == 0:
            raise ValueError("Cannot run classifier on an empty face crop.")

        face_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(face_rgb)
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            logits = self.model(image_tensor)
            probs = torch.softmax(logits, dim=1)

        prob_live = float(probs[0, 0].detach().cpu().item())
        prob_spoof = float(probs[0, 1].detach().cpu().item())
        confidence = max(prob_live, prob_spoof)
        pred_name = self.classify_spoof_probability(prob_spoof, classifier_overrides)

        return FASPrediction(
            pred_name=pred_name,
            prob_live=prob_live,
            prob_spoof=prob_spoof,
            confidence=confidence,
        )
