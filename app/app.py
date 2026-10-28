#!/usr/bin/env python3
"""NeuroVision web app (Flask) — Brain Tumor MRI classifier interface.

Run:
    python app/app.py                      # http://127.0.0.1:5000

Pages:
  /          upload an MRI image, see prediction + Grad-CAM heatmap
  /metrics   test-set metrics from results/test_metrics.json
  POST /api/predict  JSON API (multipart field "image")

Every prediction comes from a real forward pass of the trained
checkpoint. If no checkpoint exists yet, the app starts anyway and
explains how to train.

Research/educational tool — not a medical diagnosis.
"""
import base64
import io
import json
import os
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.infer import CALIBRATION_NOTE, BrainPredictor  # noqa: E402

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB upload limit
ALLOWED_EXTS = {"png", "jpg", "jpeg", "bmp", "webp"}

CHECKPOINT = Path(os.environ.get(
    "NEUROVISION_CHECKPOINT",
    REPO / "checkpoints" / "neurovision_resnet18_best.pt"))
METRICS_JSON = REPO / "results" / "test_metrics.json"

DISCLAIMER = ("Research/educational tool — not a medical diagnosis. "
              "Only a qualified clinician with the full patient history "
              "and original scans can make a diagnosis.")

_predictor = None
_predictor_error = None


def get_predictor():
    """Lazily load the model; never crash the app if training hasn't run."""
    global _predictor, _predictor_error
    if _predictor is None and _predictor_error is None:
        try:
            _predictor = BrainPredictor(CHECKPOINT)
        except Exception as e:  # noqa: BLE001
            _predictor_error = str(e)
    return _predictor


def pil_to_data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def load_metrics():
    if METRICS_JSON.exists():
        return json.loads(METRICS_JSON.read_text())
    return None


@app.route("/")
def index():
    predictor = get_predictor()
    return render_template("index.html",
                           model_ready=predictor is not None,
                           model_error=_predictor_error,
                           disclaimer=DISCLAIMER)


@app.route("/metrics")
def metrics():
    return render_template("metrics.html", metrics=load_metrics(),
                           disclaimer=DISCLAIMER)


@app.route("/predict", methods=["POST"])
def predict_form():
    predictor = get_predictor()
    if predictor is None:
        return render_template("index.html", model_ready=False,
                               model_error=_predictor_error,
                               disclaimer=DISCLAIMER), 503
    file = request.files.get("image")
    if file is None or not file.filename:
        return render_template("index.html", model_ready=True,
                               error="No file selected. Please choose an MRI image.",
                               disclaimer=DISCLAIMER), 400
    try:
        image = Image.open(file.stream)
        result = predictor.predict(image)
    except Exception as e:  # noqa: BLE001
        return render_template("index.html", model_ready=True,
                               error=f"Could not read that image: {e}",
                               disclaimer=DISCLAIMER), 400
    return render_template(
        "result.html",
        result=result,
        input_uri=pil_to_data_uri(result["input_image"]),
        overlay_uri=pil_to_data_uri(result["overlay"]),
        calibration_note=CALIBRATION_NOTE,
        disclaimer=DISCLAIMER)


@app.route("/api/predict", methods=["POST"])
def predict_api():
    predictor = get_predictor()
    if predictor is None:
        return jsonify({"error": "model not trained yet",
                        "detail": _predictor_error}), 503
    file = request.files.get("image")
    if file is None or not file.filename:
        return jsonify({"error": "multipart field 'image' is required"}), 400
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXTS:
        return jsonify({"error": f"unsupported file type '.{ext}'"}), 400
    try:
        result = predictor.predict(Image.open(file.stream))
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"could not read image: {e}"}), 400
    return jsonify({
        "predicted_class": result["predicted_class"],
        "predicted_display": result["predicted_display"],
        "confidence": result["confidence"],
        "probabilities": result["probabilities"],
        "gradcam_overlay_png_base64": base64.b64encode(
            _png_bytes(result["overlay"])).decode(),
        "note": CALIBRATION_NOTE,
        "disclaimer": DISCLAIMER,
    })


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_ready": get_predictor() is not None})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
