"""
predict.py
Loads the trained bottle classifier and predicts good/defective on new images.

Usage:
    python3 predict.py path/to/image1.png path/to/image2.png ...
"""

import sys
import joblib
import numpy as np

from features import extract_features

MODEL_PATH = "/home/claude/bottle_project/bottle_classifier.joblib"


def load_model():
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["scaler"], bundle["model_name"]


def predict_image(path, model, scaler):
    feat = extract_features(path).reshape(1, -1)
    feat_s = scaler.transform(feat)
    pred = model.predict(feat_s)[0]
    label = "DEFECTIVE" if pred == 1 else "GOOD"

    # confidence, if the model supports probability estimates
    conf = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(feat_s)[0]
        conf = proba[pred]

    return label, conf


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 predict.py path/to/image1.png [image2.png ...]")
        sys.exit(1)

    model, scaler, model_name = load_model()
    print(f"Loaded model: {model_name}\n")

    for path in sys.argv[1:]:
        try:
            label, conf = predict_image(path, model, scaler)
            conf_str = f" (confidence: {conf:.2f})" if conf is not None else ""
            print(f"{path}: {label}{conf_str}")
        except Exception as e:
            print(f"{path}: ERROR - {e}")


if __name__ == "__main__":
    main()
