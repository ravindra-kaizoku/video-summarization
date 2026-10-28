"""
training/preprocess.py
=======================
Pre-processing pipeline:
  - If the Kaggle dataset is available: loads real SumMe/TVSum annotations.
  - Falls back to synthetic benchmark data only when the dataset has not been
    downloaded (useful for CI / quick sanity checks).

Usage (from project root):
    python -m training.preprocess
"""

import os
import json
import numpy as np
import training.config as cfg


def generate_benchmark_dataset(
    output_path: str = None,
    num_summe_videos: int = 25,
    num_tvsum_videos: int = 50,
    seq_len_range: tuple = (10, 30),
):
    """
    Generates realistic synthetic TVSum/SumMe benchmark feature files
    with visual features (512-dim), text features (384-dim), and
    ground-truth human annotations [0, 1].

    NOTE: This function is the FALLBACK when real videos are unavailable.
    To use REAL annotations, run training/feature_extraction.py first.
    """
    output_path = output_path or cfg.PROCESSED_DATASET_PATH
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dataset = []
    np.random.seed(cfg.SEED)

    # SumMe videos
    for i in range(1, num_summe_videos + 1):
        n = np.random.randint(*seq_len_range)
        v = np.random.randn(n, 512).astype(np.float32)
        v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
        t = np.random.randn(n, 384).astype(np.float32)
        t /= (np.linalg.norm(t, axis=1, keepdims=True) + 1e-8)

        # Realistic multi-modal importance signal
        v_sig = np.std(v, axis=1) * 5.0
        t_sig = np.sum(np.abs(t[:, :20]), axis=1)
        raw   = (0.3
                 + 0.35 * v_sig / (v_sig.max() + 1e-6)
                 + 0.30 * t_sig / (t_sig.max() + 1e-6))
        # Add temporal structure: beginning/end slightly more important
        temporal_bias = np.abs(np.linspace(-0.1, 0.1, n))
        scores = np.clip(raw + temporal_bias + np.random.normal(0, 0.03, n), 0.10, 0.95)

        dataset.append({
            "video_id": f"summe_video_{i:02d}",
            "dataset_type": "SumMe",
            "num_segments": n,
            "visual_features": v.tolist(),
            "text_features": t.tolist(),
            "importance_scores": scores.tolist(),
        })

    # TVSum videos
    for i in range(1, num_tvsum_videos + 1):
        n = np.random.randint(*seq_len_range)
        v = np.random.randn(n, 512).astype(np.float32)
        v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
        t = np.random.randn(n, 384).astype(np.float32)
        t /= (np.linalg.norm(t, axis=1, keepdims=True) + 1e-8)

        v_sig = np.std(v, axis=1) * 5.0
        t_sig = np.sum(np.abs(t[:, 20:40]), axis=1)
        raw   = (0.35
                 + 0.30 * v_sig / (v_sig.max() + 1e-6)
                 + 0.30 * t_sig / (t_sig.max() + 1e-6))
        temporal_bias = np.abs(np.linspace(-0.1, 0.1, n))
        scores = np.clip(raw + temporal_bias + np.random.normal(0, 0.03, n), 0.10, 0.95)

        dataset.append({
            "video_id": f"tvsum_video_{i:02d}",
            "dataset_type": "TVSum",
            "num_segments": n,
            "visual_features": v.tolist(),
            "text_features": t.tolist(),
            "importance_scores": scores.tolist(),
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    print(f"[Preprocess] Synthetic benchmark dataset: {len(dataset)} videos -> {output_path}")
    return output_path


def load_or_generate_dataset(output_path: str = None):
    """
    Returns the path to a ready-to-use processed_dataset.json.
    - If datasets/processed_dataset.json already exists, returns it directly.
    - Otherwise generates synthetic benchmark data.
    - To use REAL data: run `python -m training.feature_extraction` first.
    """
    output_path = output_path or cfg.PROCESSED_DATASET_PATH

    if os.path.exists(output_path):
        with open(output_path, encoding="utf-8") as f:
            data = json.load(f)
        print(f"[Preprocess] Loaded existing dataset: {len(data)} videos from {output_path}")
        return output_path

    print("[Preprocess] No dataset found. Generating synthetic benchmark data...")
    return generate_benchmark_dataset(output_path)


if __name__ == "__main__":
    load_or_generate_dataset()
