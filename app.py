"""
app.py
Gradio web app for the bottle defect classifier.

Run locally with:
    python3 app.py

Then open the local URL it prints (usually http://127.0.0.1:7860).
Requires: gradio, joblib, scikit-learn, scikit-image, pillow, numpy
    pip install gradio joblib scikit-learn scikit-image pillow numpy
"""

import os
import tempfile

import joblib
import numpy as np
import gradio as gr
from PIL import Image

from features import extract_features, IMG_SIZE

MODEL_PATH = "bottle_classifier.joblib"  # place this file alongside app.py

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
scaler = bundle["scaler"]
model_name = bundle["model_name"]


def predict(image: Image.Image):
    if image is None:
        return "No image provided", None

    # Save to a temp path since our feature extractor works off a file path.
    # Uses the OS's own temp folder so this works on Windows, Mac, and Linux.
    tmp_dir = tempfile.gettempdir()
    tmp_path = os.path.join(tmp_dir, "_gradio_upload.png")
    image.convert("RGB").resize((IMG_SIZE, IMG_SIZE)).save(tmp_path)

    feat = extract_features(tmp_path).reshape(1, -1)
    feat_s = scaler.transform(feat)
    pred = model.predict(feat_s)[0]

    label = "DEFECTIVE" if pred == 1 else "GOOD"

    conf_dict = {"GOOD": 0.0, "DEFECTIVE": 0.0}
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(feat_s)[0]
        conf_dict = {"GOOD": float(proba[0]), "DEFECTIVE": float(proba[1])}

    verdict = f"## {'✅ GOOD' if label == 'GOOD' else '❌ DEFECTIVE'}"
    return verdict, conf_dict


with gr.Blocks(title="Bottle Defect Classifier") as demo:
    gr.Markdown("# 🍾 Bottle Defect Classifier")
    gr.Markdown(
        f"Upload a photo of a bottle and the model (**{model_name}**) will tell you "
        "whether it looks good or defective (broken / contaminated)."
    )

    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="pil", label="Upload bottle image")
            submit_btn = gr.Button("Check bottle", variant="primary")
        with gr.Column():
            verdict_output = gr.Markdown(label="Result")
            confidence_output = gr.Label(label="Confidence", num_top_classes=2)

    submit_btn.click(
        fn=predict, inputs=img_input, outputs=[verdict_output, confidence_output]
    )
    img_input.change(
        fn=predict, inputs=img_input, outputs=[verdict_output, confidence_output]
    )

    gr.Markdown(
        "*Model trained on the MVTec AD bottle dataset "
        "(HOG + color histogram features, SVM classifier, CPU-only).*"
    )


if __name__ == "__main__":
    demo.launch()
