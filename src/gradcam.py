"""Grad-CAM visualization for the NeuroVision classifier.

Reference: R. R. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep
Networks via Gradient-based Localization," ICCV 2017.

Given a trained classifier and an input image, Grad-CAM produces a heatmap
highlighting the image regions that most influenced the predicted (or a
chosen) class. This is a *post-hoc explanation* of the model, not an
independent tumor detector and not a segmentation of the tumor.
"""
import matplotlib
import numpy as np
import torch
from PIL import Image


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._activations = None
        self._gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self._activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def heatmap(self, x: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        """Return a HxW heatmap in [0, 1] for input batch x (uses x[0])."""
        self.model.eval()
        self.model.zero_grad()
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits[0].argmax())
        score = logits[0, class_idx]
        score.backward()

        weights = self._gradients[0].mean(dim=(1, 2), keepdim=True)  # C x 1 x 1
        cam = (weights * self._activations[0]).sum(dim=0)            # H x W
        cam = torch.relu(cam)
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        h, w = x.shape[2], x.shape[3]
        cam_img = Image.fromarray((cam.cpu().numpy() * 255).astype(np.uint8))
        cam_img = cam_img.resize((w, h), Image.BILINEAR)
        return (np.asarray(cam_img, dtype=np.float32) / 255.0)

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        return self.heatmap(x, class_idx)


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray,
                    alpha: float = 0.45) -> np.ndarray:
    """Blend a [0,1] heatmap (HxW) over an RGB uint8 image -> RGB uint8 overlay."""
    if image.dtype != np.uint8:
        raise ValueError("image must be uint8")
    h, w = image.shape[:2]
    if heatmap.shape != (h, w):
        hm = Image.fromarray((heatmap * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
        heatmap = np.asarray(hm, dtype=np.float32) / 255.0
    cmap = matplotlib.colormaps["jet"]
    colored = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    blended = (alpha * colored + (1 - alpha) * image).astype(np.uint8)
    return blended
