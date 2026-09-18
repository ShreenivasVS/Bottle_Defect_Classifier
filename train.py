"""
train.py
Trains and compares Random Forest and SVM classifiers on extracted features,
selects the best model using the validation set, then reports final
performance on the held-out test set.
"""

import json
import time
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

from features import extract_batch

SPLITS_PATH = "/home/claude/bottle_project/splits.json"
MODEL_PATH = "/home/claude/bottle_project/bottle_classifier.joblib"
REPORT_PATH = "/home/claude/bottle_project/evaluation_report.md"


def load_splits():
    with open(SPLITS_PATH) as f:
        return json.load(f)


def evaluate(model, scaler, X, y, name=""):
    Xs = scaler.transform(X)
    preds = model.predict(Xs)
    acc = accuracy_score(y, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y, preds, average=None, labels=[0, 1], zero_division=0
    )
    cm = confusion_matrix(y, preds, labels=[0, 1])
    return {
        "name": name,
        "accuracy": acc,
        "precision_good": prec[0], "recall_good": rec[0], "f1_good": f1[0],
        "precision_defective": prec[1], "recall_defective": rec[1], "f1_defective": f1[1],
        "confusion_matrix": cm.tolist(),
        "preds": preds,
    }


def print_eval(ev):
    print(f"\n--- {ev['name']} ---")
    print(f"Accuracy: {ev['accuracy']:.3f}")
    print(f"Defective class -> precision: {ev['precision_defective']:.3f}  "
          f"recall: {ev['recall_defective']:.3f}  f1: {ev['f1_defective']:.3f}")
    print(f"Good class      -> precision: {ev['precision_good']:.3f}  "
          f"recall: {ev['recall_good']:.3f}  f1: {ev['f1_good']:.3f}")
    print("Confusion matrix [rows=true, cols=pred] order=[good, defective]:")
    print(np.array(ev["confusion_matrix"]))


def main():
    splits = load_splits()

    print("Extracting features for train set...")
    X_train, y_train = extract_batch(splits["train"])
    print("Extracting features for val set...")
    X_val, y_val = extract_batch(splits["val"])
    print("Extracting features for test set...")
    X_test, y_test = extract_batch(splits["test"])

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)

    candidates = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=None, class_weight="balanced",
            random_state=42, n_jobs=-1,
        ),
        "SVM_RBF": SVC(
            kernel="rbf", C=10, gamma="scale", class_weight="balanced",
            probability=True, random_state=42,
        ),
    }

    # In defect detection, missing a real defect (false negative) is usually worse
    # than a false alarm -> selection criterion favors recall on the defective
    # class (F-beta with beta=2), not plain F1.
    def f_beta(precision, recall, beta=2):
        if precision == 0 and recall == 0:
            return 0.0
        b2 = beta ** 2
        return (1 + b2) * precision * recall / (b2 * precision + recall + 1e-12)

    trained = {}
    best_name, best_model, best_score = None, None, -1
    for name, clf in candidates.items():
        t0 = time.time()
        clf.fit(X_train_s, y_train)
        elapsed = time.time() - t0
        ev = evaluate(clf, scaler, X_val, y_val, name=f"{name} (val)")
        ev["f2_defective"] = f_beta(ev["precision_defective"], ev["recall_defective"])
        print_eval(ev)
        print(f"F2 (defective, recall-weighted): {ev['f2_defective']:.3f}")
        print(f"(trained in {elapsed:.1f}s)")
        trained[name] = clf
        if ev["f2_defective"] > best_score:
            best_score = ev["f2_defective"]
            best_name = name
            best_model = clf

    print(f"\n=== Best model on validation (by recall-weighted F2 on defective class): {best_name} ===")

    # Final evaluation on held-out test set -- show ALL candidates for transparency
    print("\n### Test-set comparison across all candidate models ###")
    all_test_evs = {}
    for name, clf in trained.items():
        ev = evaluate(clf, scaler, X_test, y_test, name=f"{name} (TEST)")
        print_eval(ev)
        all_test_evs[name] = ev

    test_ev = all_test_evs[best_name]
    print(f"\n=== Final selected model: {best_name} ===")
    print("\n" + classification_report(
        y_test, test_ev["preds"], target_names=["good", "defective"], zero_division=0
    ))

    # Save model + scaler bundled together
    joblib.dump({"model": best_model, "scaler": scaler, "model_name": best_name}, MODEL_PATH)
    print(f"\nSaved trained model to {MODEL_PATH}")

    # Write evaluation report
    with open(REPORT_PATH, "w") as f:
        f.write("# Bottle Defect Classifier — Evaluation Report\n\n")
        f.write(f"**Best model:** {best_name}\n\n")
        f.write(f"**Test set size:** {len(y_test)} images ")
        f.write(f"({sum(1 for v in y_test if v == 0)} good, {sum(1 for v in y_test if v == 1)} defective)\n\n")
        f.write(f"**Test accuracy:** {test_ev['accuracy']:.3f}\n\n")
        f.write("## Defective class (the one that matters most)\n")
        f.write(f"- Precision: {test_ev['precision_defective']:.3f}\n")
        f.write(f"- Recall: {test_ev['recall_defective']:.3f}\n")
        f.write(f"- F1: {test_ev['f1_defective']:.3f}\n\n")
        f.write("## Good class\n")
        f.write(f"- Precision: {test_ev['precision_good']:.3f}\n")
        f.write(f"- Recall: {test_ev['recall_good']:.3f}\n")
        f.write(f"- F1: {test_ev['f1_good']:.3f}\n\n")
        f.write("## Confusion matrix (rows=true, cols=predicted; order=[good, defective])\n")
        f.write("```\n")
        f.write(str(np.array(test_ev["confusion_matrix"])))
        f.write("\n```\n")
    print(f"Saved evaluation report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
