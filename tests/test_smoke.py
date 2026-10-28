"""Smoke tests: model builds, forward pass works, checkpoint (if present)
loads and predicts. Run: python -m pytest tests/ -q"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_brain import CLASS_NAMES, DISPLAY_NAMES  # noqa: E402
from src.models import NeuroNet  # noqa: E402

CKPT = Path(__file__).resolve().parent.parent / "checkpoints" / "neurovision_resnet18_best.pt"


def test_class_names():
    assert CLASS_NAMES == ["glioma", "meningioma", "notumor", "pituitary"]
    assert DISPLAY_NAMES["notumor"] == "No Tumor"


def test_forward_shape():
    model = NeuroNet(num_classes=4, pretrained=False)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 4)


def test_gradcam_runs():
    from src.gradcam import GradCAM
    model = NeuroNet(num_classes=4, pretrained=False)
    cam = GradCAM(model, model.layer4)
    heat = cam(torch.randn(1, 3, 224, 224))
    assert heat.shape == (224, 224)
    assert heat.min() >= 0.0 and heat.max() <= 1.0


def test_checkpoint_predicts_if_present():
    if not CKPT.exists():
        return  # training has not run in this checkout; forward test covers the path
    from PIL import Image
    from src.infer import BrainPredictor
    predictor = BrainPredictor(CKPT)
    result = predictor.predict(Image.new("RGB", (300, 300), color=90))
    assert result["predicted_class"] in CLASS_NAMES
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-4
