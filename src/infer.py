"""Single-image inference + Grad-CAM for NeuroVision.

Used by the Flask app and by the CLI:

    python -m src.infer data/raw/Testing/glioma/Te-gl_1.jpg
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.data_brain import CLASS_NAMES, DISPLAY_NAMES, eval_transforms
from src.gradcam import GradCAM, overlay_heatmap
from src.models import NeuroNet

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CKPT = REPO_ROOT / "checkpoints" / "neurovision_resnet18_best.pt"

CALIBRATION_NOTE = ("Scores are the model's softmax outputs. They show how the "
                    "model distributes its confidence across the four classes; "
                    "they are not calibrated medical probabilities.")


class BrainPredictor:
    def __init__(self, checkpoint=DEFAULT_CKPT):
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model = NeuroNet(num_classes=4, pretrained=False)
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.gradcam = GradCAM(self.model, self.model.layer4)
        self.tf = eval_transforms(224)
        self.checkpoint_epoch = ck.get("epoch")

    def predict(self, image: Image.Image) -> dict:
        img = image.convert("RGB").resize((224, 224))
        x = self.tf(img).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1)[0].numpy()
        pred = int(probs.argmax())
        heat = self.gradcam(x, pred)
        overlay = Image.fromarray(overlay_heatmap(np.asarray(img), heat))
        return {
            "predicted_class": CLASS_NAMES[pred],
            "predicted_display": DISPLAY_NAMES[CLASS_NAMES[pred]],
            "confidence": float(probs[pred]),
            "probabilities": {DISPLAY_NAMES[c]: float(probs[i])
                              for i, c in enumerate(CLASS_NAMES)},
            "overlay": overlay,
            "input_image": img,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    args = ap.parse_args()
    predictor = BrainPredictor(args.checkpoint)
    result = predictor.predict(Image.open(args.image))
    print(f"prediction: {result['predicted_display']} "
          f"({result['confidence']:.3f})")
    for name, p in result["probabilities"].items():
        print(f"  {name:16s} {p:.4f}")
    print(CALIBRATION_NOTE)


if __name__ == "__main__":
    main()
