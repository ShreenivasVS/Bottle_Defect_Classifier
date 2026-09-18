# Bottle Defect Classifier — Gradio App

## Setup

1. Install dependencies:
   ```
   pip install gradio joblib scikit-learn scikit-image pillow numpy
   ```

2. Make sure these files are all in the same folder:
   - `app.py`
   - `features.py`
   - `bottle_classifier.joblib`

3. Run it:
   ```
   python3 app.py
   ```

4. Open the URL it prints in your browser (usually `http://127.0.0.1:7860`).

## Using it

Upload a photo of a bottle, and the app will show:
- A GOOD / DEFECTIVE verdict
- A confidence breakdown between both classes

## Files in this bundle

- `prepare_data.py` — builds the labeled dataset + train/val/test split
- `features.py` — extracts HOG + color histogram features (needed by both training and the app)
- `train.py` — trains and evaluates the classifier
- `predict.py` — command-line inference on individual images
- `app.py` — the Gradio web app (visual interface)
- `bottle_classifier.joblib` — the trained model
- `evaluation_report.md` — test-set performance metrics
