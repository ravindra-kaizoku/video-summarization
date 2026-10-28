"""
training/evaluate.py
====================
Evaluates the trained video importance model on held-out test videos.

Usage (from project root):
    python -m training.evaluate

Metrics reported:
  - MSE (Mean Squared Error)
  - MAE (Mean Absolute Error)
  - Pearson correlation coefficient
  - Score variance check (detects degenerate flat predictions)
  - Per-video predicted vs ground-truth score display
"""

import os
import json
import torch
import numpy as np
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

from training.model   import MultimodalImportanceModel
from training.dataset import MultimodalVideoDataset
import training.config as cfg


def evaluate_model(
    model_path: str  = None,
    dataset_path: str = None,
    split_path: str   = None,
    output_path: str  = None,
):
    model_path   = model_path   or cfg.MODEL_SAVE_PATH
    dataset_path = dataset_path or cfg.PROCESSED_DATASET_PATH
    split_path   = split_path   or cfg.DATA_SPLIT_PATH
    output_path  = output_path  or cfg.EVAL_OUTPUT_PATH

    print("=== EVALUATING VIDEO IMPORTANCE MODEL ===")

    # Validate files
    for p, name in [(model_path, "model"), (dataset_path, "dataset"), (split_path, "split")]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required {name} file not found: {p}")

    # Load test split
    with open(split_path, encoding="utf-8") as f:
        split = json.load(f)
    test_ids = set(split.get("test_videos", []))

    with open(dataset_path, encoding="utf-8") as f:
        all_vids = json.load(f)

    test_vids = [v for v in all_vids if v["video_id"] in test_ids]
    if not test_vids:
        print("[WARN] No test videos matched split file. Using last 12 videos as fallback.")
        test_vids = all_vids[-12:]

    print(f"Evaluating on {len(test_vids)} test videos...")

    # Load model
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = MultimodalImportanceModel(
        visual_dim=cfg.VISUAL_DIM,
        text_dim=cfg.TEXT_DIM,
        fused_dim=cfg.FUSED_DIM,
        hidden_dim=cfg.HIDDEN_DIM,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: epoch={checkpoint.get('epoch','?')} | val_loss={checkpoint.get('val_loss','?'): .5f}")

    loader = DataLoader(MultimodalVideoDataset(test_vids, cfg.MAX_SEQ_LEN), batch_size=1, shuffle=False)

    all_preds, all_tgts, per_video = [], [], []

    with torch.no_grad():
        for i, (vid, batch) in enumerate(zip(test_vids, loader)):
            v    = batch["visual_features"]
            t    = batch["text_features"]
            tgts = batch["importance_scores"].squeeze(-1).numpy()[0]
            slen = batch["seq_len"].item()

            preds = model(v, t).squeeze(-1).numpy()[0]

            p_valid = preds[:slen]
            t_valid = tgts[:slen]

            all_preds.extend(p_valid)
            all_tgts.extend(t_valid)

            per_video.append({
                "video_id"     : vid["video_id"],
                "segments"     : slen,
                "pred_min"     : float(p_valid.min()),
                "pred_max"     : float(p_valid.max()),
                "pred_mean"    : float(p_valid.mean()),
                "pred_std"     : float(p_valid.std()),
                "gt_min"       : float(t_valid.min()),
                "gt_max"       : float(t_valid.max()),
                "gt_mean"      : float(t_valid.mean()),
                "sample_preds" : [round(float(x), 3) for x in p_valid[:8]],
            })

    all_preds = np.array(all_preds)
    all_tgts  = np.array(all_tgts)

    mse  = float(np.mean((all_preds - all_tgts) ** 2))
    mae  = float(np.mean(np.abs(all_preds - all_tgts)))

    if all_preds.std() > 1e-6 and all_tgts.std() > 1e-6:
        pearson_r, pearson_p = pearsonr(all_preds, all_tgts)
    else:
        pearson_r, pearson_p = 0.0, 1.0

    pred_variance = float(all_preds.var())
    flat_warning  = pred_variance < 0.001  # degenerate flat prediction check

    # Binary metrics (threshold 0.5)
    bp = (all_preds >= 0.5).astype(int)
    bt = (all_tgts  >= 0.5).astype(int)
    tp = int(np.sum((bp == 1) & (bt == 1)))
    fp = int(np.sum((bp == 1) & (bt == 0)))
    fn = int(np.sum((bp == 0) & (bt == 1)))
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    results = {
        "model_path"          : model_path,
        "model_epoch"         : checkpoint.get("epoch"),
        "model_val_loss"      : checkpoint.get("val_loss"),
        "test_videos_count"   : len(test_vids),
        "total_test_segments" : len(all_preds),
        "metrics": {
            "mse"       : round(mse, 6),
            "mae"       : round(mae, 6),
            "pearson_r" : round(float(pearson_r), 4),
            "pearson_p" : round(float(pearson_p), 4),
            "precision" : round(precision, 4),
            "recall"    : round(recall, 4),
            "f1_score"  : round(f1, 4),
            "pred_variance": round(pred_variance, 6),
            "flat_prediction_warning": flat_warning,
        },
        "per_video_results": per_video,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # --- Console report ---
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"  MSE          : {mse:.6f}")
    print(f"  MAE          : {mae:.6f}")
    print(f"  Pearson r    : {pearson_r:.4f}  (p={pearson_p:.4f})")
    print(f"  Precision    : {precision:.4f}")
    print(f"  Recall       : {recall:.4f}")
    print(f"  F1 Score     : {f1:.4f}")
    print(f"  Pred variance: {pred_variance:.6f}")
    if flat_warning:
        print("  [WARNING] Predictions are nearly flat — model may not be well-trained!")
    print("\nPer-video sample predictions (first 8 segments):")
    for pv in per_video[:5]:
        print(f"  {pv['video_id']} | std={pv['pred_std']:.3f} | "
              f"preds: {pv['sample_preds']}")
    print(f"\nFull results saved: {output_path}")
    return results


if __name__ == "__main__":
    evaluate_model()
