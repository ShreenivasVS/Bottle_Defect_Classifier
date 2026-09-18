# Bottle Defect Classifier — Evaluation Report

**Best model:** SVM_RBF

**Test set size:** 44 images (35 good, 9 defective)

**Test accuracy:** 0.977

## Defective class (the one that matters most)
- Precision: 1.000
- Recall: 0.889
- F1: 0.941

## Good class
- Precision: 0.972
- Recall: 1.000
- F1: 0.986

## Confusion matrix (rows=true, cols=predicted; order=[good, defective])
```
[[35  0]
 [ 1  8]]
```
