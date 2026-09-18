"""
prepare_data.py
Pools all MVTec 'bottle' images into a single labeled list (good=0, defective=1)
and creates a stratified train/val/test split.
"""

import os
import json
from sklearn.model_selection import train_test_split

DATA_ROOT = "/home/claude/bottle_data/bottle"  # change this to your local path
OUTPUT_JSON = "/home/claude/bottle_project/splits.json"

RANDOM_SEED = 42


def build_labeled_list(root):
    data = []

    # Good images live in both train/good and test/good
    for split in ["train", "test"]:
        good_dir = os.path.join(root, split, "good")
        if os.path.isdir(good_dir):
            for fname in sorted(os.listdir(good_dir)):
                data.append({"path": os.path.join(good_dir, fname), "label": 0})

    # Defective images: every other subfolder of test/
    test_dir = os.path.join(root, "test")
    for sub in sorted(os.listdir(test_dir)):
        if sub == "good":
            continue
        sub_dir = os.path.join(test_dir, sub)
        if os.path.isdir(sub_dir):
            for fname in sorted(os.listdir(sub_dir)):
                data.append({
                    "path": os.path.join(sub_dir, fname),
                    "label": 1,
                    "defect_type": sub,
                })

    return data


def stratified_split(data, train_frac=0.70, val_frac=0.15, test_frac=0.15):
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6

    labels = [d["label"] for d in data]
    train, temp = train_test_split(
        data, train_size=train_frac, stratify=labels, random_state=RANDOM_SEED
    )
    temp_labels = [d["label"] for d in temp]
    val_size = val_frac / (val_frac + test_frac)
    val, test = train_test_split(
        temp, train_size=val_size, stratify=temp_labels, random_state=RANDOM_SEED
    )
    return train, val, test


def summarize(name, subset):
    good = sum(1 for d in subset if d["label"] == 0)
    bad = sum(1 for d in subset if d["label"] == 1)
    print(f"{name:6s}: total={len(subset):3d}  good={good:3d}  defective={bad:3d}")


if __name__ == "__main__":
    data = build_labeled_list(DATA_ROOT)
    train, val, test = stratified_split(data)

    summarize("train", train)
    summarize("val", val)
    summarize("test", test)

    with open(OUTPUT_JSON, "w") as f:
        json.dump({"train": train, "val": val, "test": test}, f, indent=2)

    print(f"\nSaved split to {OUTPUT_JSON}")
