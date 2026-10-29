"""
research_evaluation/build_dataset_audit.py
==========================================
Audits the existing dataset files, uploads, and caches to produce:
1. research_evaluation/dataset_audit.csv
2. research_evaluation/dataset_summary.md
"""

import os
import json
import csv
import glob
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES_DIR = os.path.join(PROJECT_ROOT, "research_evaluation")

# 1. Audit processed_dataset.json
with open(os.path.join(PROJECT_ROOT, "datasets", "processed_dataset.json"), encoding="utf-8") as f:
    processed_data = json.load(f)

# Audit data_split.json
with open(os.path.join(PROJECT_ROOT, "datasets", "data_split.json"), encoding="utf-8") as f:
    split = json.load(f)

train_set = set(split.get("train_videos", []))
val_set = set(split.get("val_videos", []))
test_set = set(split.get("test_videos", []))

csv_path = os.path.join(RES_DIR, "dataset_audit.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "video_id", "dataset_type", "split", "num_segments",
        "visual_dim", "text_dim", "score_min", "score_max",
        "score_mean", "score_std", "source_type"
    ])
    for item in processed_data:
        vid = item["video_id"]
        dtype = item["dataset_type"]
        split_name = "train" if vid in train_set else "val" if vid in val_set else "test" if vid in test_set else "unassigned"
        segs = item["num_segments"]
        v_dim = len(item["visual_features"][0]) if item["visual_features"] else 0
        t_dim = len(item["text_features"][0]) if item["text_features"] else 0
        scores = np.array(item["importance_scores"])
        writer.writerow([
            vid, dtype, split_name, segs, v_dim, t_dim,
            f"{scores.min():.4f}", f"{scores.max():.4f}",
            f"{scores.mean():.4f}", f"{scores.std():.4f}",
            "synthetic_random_gaussian"
        ])

print(f"Generated: {csv_path} ({len(processed_data)} entries)")
