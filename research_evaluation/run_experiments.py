"""
research_evaluation/run_experiments.py
======================================
Complete experimental evaluation suite for the video summarization project.
Runs all feasible experiments and saves results.

Usage (from project root):
    python -m research_evaluation.run_experiments
"""

import os
import sys
import json
import time
import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from torch.utils.data import DataLoader
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from training.model import MultimodalImportanceModel
from training.dataset import MultimodalVideoDataset
import training.config as cfg

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "research_evaluation", "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data_and_model():
    """Load test data, model, and data split."""
    with open(cfg.PROCESSED_DATASET_PATH, encoding="utf-8") as f:
        all_videos = json.load(f)

    with open(cfg.DATA_SPLIT_PATH, encoding="utf-8") as f:
        split = json.load(f)

    test_ids = set(split.get("test_videos", []))
    train_ids = set(split.get("train_videos", []))
    val_ids = set(split.get("val_videos", []))

    test_vids = [v for v in all_videos if v["video_id"] in test_ids]
    train_vids = [v for v in all_videos if v["video_id"] in train_ids]
    val_vids = [v for v in all_videos if v["video_id"] in val_ids]

    # Load both checkpoints
    models = {}
    for name, path in [("pt", "models/video_importance_model.pt"),
                        ("pth", "models/video_importance_model.pth")]:
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            model = MultimodalImportanceModel(
                visual_dim=cfg.VISUAL_DIM, text_dim=cfg.TEXT_DIM,
                fused_dim=cfg.FUSED_DIM, hidden_dim=cfg.HIDDEN_DIM
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            models[name] = {
                "model": model,
                "checkpoint": checkpoint,
                "path": path
            }

    return all_videos, train_vids, val_vids, test_vids, models, split


def compute_metrics(preds, targets):
    """Compute comprehensive evaluation metrics."""
    preds = np.array(preds)
    targets = np.array(targets)

    mse = float(np.mean((preds - targets) ** 2))
    mae = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(mse))

    # Correlation metrics
    if preds.std() > 1e-6 and targets.std() > 1e-6:
        pearson_r, pearson_p = pearsonr(preds, targets)
        spearman_r, spearman_p = spearmanr(preds, targets)
        kendall_t, kendall_p = kendalltau(preds, targets)
    else:
        pearson_r = pearson_p = 0.0
        spearman_r = spearman_p = 0.0
        kendall_t = kendall_p = 0.0

    # Binary classification metrics at threshold 0.5
    bp = (preds >= 0.5).astype(int)
    bt = (targets >= 0.5).astype(int)
    tp = int(np.sum((bp == 1) & (bt == 1)))
    fp = int(np.sum((bp == 1) & (bt == 0)))
    fn = int(np.sum((bp == 0) & (bt == 1)))
    tn = int(np.sum((bp == 0) & (bt == 0)))
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-8)

    return {
        "mse": round(mse, 6),
        "rmse": round(rmse, 6),
        "mae": round(mae, 6),
        "pearson_r": round(float(pearson_r), 4),
        "pearson_p": round(float(pearson_p), 4),
        "spearman_r": round(float(spearman_r), 4),
        "spearman_p": round(float(spearman_p), 4),
        "kendall_tau": round(float(kendall_t), 4),
        "kendall_p": round(float(kendall_p), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "pred_mean": round(float(preds.mean()), 4),
        "pred_std": round(float(preds.std()), 4),
        "pred_min": round(float(preds.min()), 4),
        "pred_max": round(float(preds.max()), 4),
        "target_mean": round(float(targets.mean()), 4),
        "target_std": round(float(targets.std()), 4),
        "target_min": round(float(targets.min()), 4),
        "target_max": round(float(targets.max()), 4),
        "n_samples": len(preds),
    }


def experiment_1_model_evaluation(test_vids, models):
    """Experiment 1: Comprehensive model evaluation on test set with both checkpoints."""
    print("\n" + "="*60)
    print("EXPERIMENT 1: Model Evaluation on Test Set")
    print("="*60)

    results = {}

    for ckpt_name, ckpt_data in models.items():
        model = ckpt_data["model"]
        checkpoint = ckpt_data["checkpoint"]

        loader = DataLoader(
            MultimodalVideoDataset(test_vids, cfg.MAX_SEQ_LEN),
            batch_size=1, shuffle=False
        )

        all_preds, all_tgts = [], []
        per_video = []

        with torch.no_grad():
            for vid, batch in zip(test_vids, loader):
                v = batch["visual_features"]
                t = batch["text_features"]
                tgts = batch["importance_scores"].squeeze(-1).numpy()[0]
                slen = batch["seq_len"].item()
                preds = model(v, t).squeeze(-1).numpy()[0]

                p_valid = preds[:slen]
                t_valid = tgts[:slen]
                all_preds.extend(p_valid)
                all_tgts.extend(t_valid)

                per_video.append({
                    "video_id": vid["video_id"],
                    "dataset_type": vid["dataset_type"],
                    "segments": slen,
                    "metrics": compute_metrics(p_valid, t_valid)
                })

        metrics = compute_metrics(all_preds, all_tgts)

        results[ckpt_name] = {
            "checkpoint_path": ckpt_data["path"],
            "epoch": checkpoint.get("epoch"),
            "val_loss": checkpoint.get("val_loss"),
            "aggregate_metrics": metrics,
            "per_video_results": per_video,
        }

        print(f"\n  Checkpoint: {ckpt_name} (epoch {checkpoint.get('epoch')})")
        print(f"  MSE: {metrics['mse']:.6f}  MAE: {metrics['mae']:.6f}  RMSE: {metrics['rmse']:.6f}")
        print(f"  Pearson r: {metrics['pearson_r']:.4f}  Spearman r: {metrics['spearman_r']:.4f}  Kendall tau: {metrics['kendall_tau']:.4f}")
        print(f"  F1: {metrics['f1_score']:.4f}  Precision: {metrics['precision']:.4f}  Recall: {metrics['recall']:.4f}")
        print(f"  Pred range: [{metrics['pred_min']:.4f}, {metrics['pred_max']:.4f}], mean={metrics['pred_mean']:.4f}, std={metrics['pred_std']:.4f}")

    save_path = os.path.join(OUTPUT_DIR, "exp1_model_evaluation.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {save_path}")
    return results


def experiment_2_per_dataset_evaluation(test_vids, models):
    """Experiment 2: Evaluate model separately on SumMe and TVSum subsets."""
    print("\n" + "="*60)
    print("EXPERIMENT 2: Per-Dataset Type Evaluation")
    print("="*60)

    model = models["pth"]["model"]  # Use latest checkpoint
    results = {}

    for dtype in ["SumMe", "TVSum"]:
        subset = [v for v in test_vids if v["dataset_type"] == dtype]
        if not subset:
            print(f"  No {dtype} videos in test set.")
            continue

        loader = DataLoader(
            MultimodalVideoDataset(subset, cfg.MAX_SEQ_LEN),
            batch_size=1, shuffle=False
        )

        all_preds, all_tgts = [], []
        with torch.no_grad():
            for vid, batch in zip(subset, loader):
                v = batch["visual_features"]
                t = batch["text_features"]
                tgts = batch["importance_scores"].squeeze(-1).numpy()[0]
                slen = batch["seq_len"].item()
                preds = model(v, t).squeeze(-1).numpy()[0]
                all_preds.extend(preds[:slen])
                all_tgts.extend(tgts[:slen])

        metrics = compute_metrics(all_preds, all_tgts)
        results[dtype] = {
            "num_videos": len(subset),
            "num_segments": len(all_preds),
            "metrics": metrics
        }
        print(f"\n  {dtype}: {len(subset)} videos, {len(all_preds)} segments")
        print(f"  MSE: {metrics['mse']:.6f}  Pearson r: {metrics['pearson_r']:.4f}  F1: {metrics['f1_score']:.4f}")

    save_path = os.path.join(OUTPUT_DIR, "exp2_per_dataset.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {save_path}")
    return results


def experiment_3_ablation_study(test_vids, models):
    """Experiment 3: Ablation study — visual-only, text-only, and multimodal."""
    print("\n" + "="*60)
    print("EXPERIMENT 3: Modality Ablation Study")
    print("="*60)

    model = models["pth"]["model"]

    loader = DataLoader(
        MultimodalVideoDataset(test_vids, cfg.MAX_SEQ_LEN),
        batch_size=1, shuffle=False
    )

    conditions = {
        "multimodal": {"zero_visual": False, "zero_text": False},
        "visual_only": {"zero_visual": False, "zero_text": True},
        "text_only": {"zero_visual": True, "zero_text": False},
        "no_input": {"zero_visual": True, "zero_text": True},
    }

    results = {}

    # Collect data first
    data_batches = []
    for vid, batch in zip(test_vids, loader):
        data_batches.append((vid, batch))

    for cond_name, cond_config in conditions.items():
        all_preds, all_tgts = [], []

        with torch.no_grad():
            for vid, batch in data_batches:
                v = batch["visual_features"].clone()
                t = batch["text_features"].clone()
                tgts = batch["importance_scores"].squeeze(-1).numpy()[0]
                slen = batch["seq_len"].item()

                if cond_config["zero_visual"]:
                    v.zero_()
                if cond_config["zero_text"]:
                    t.zero_()

                preds = model(v, t).squeeze(-1).numpy()[0]
                all_preds.extend(preds[:slen])
                all_tgts.extend(tgts[:slen])

        metrics = compute_metrics(all_preds, all_tgts)
        results[cond_name] = {
            "condition": cond_config,
            "metrics": metrics,
        }
        print(f"\n  {cond_name}:")
        print(f"    MSE: {metrics['mse']:.6f}  Pearson r: {metrics['pearson_r']:.4f}")
        print(f"    Pred range: [{metrics['pred_min']:.4f}, {metrics['pred_max']:.4f}], std={metrics['pred_std']:.4f}")

    save_path = os.path.join(OUTPUT_DIR, "exp3_ablation.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {save_path}")
    return results


def experiment_4_cross_split_evaluation(all_videos, models, split):
    """Experiment 4: Evaluate on train, val, and test splits to check overfitting."""
    print("\n" + "="*60)
    print("EXPERIMENT 4: Cross-Split Generalization Analysis")
    print("="*60)

    model = models["pth"]["model"]
    results = {}

    for split_name, ids_key in [("train", "train_videos"), ("val", "val_videos"), ("test", "test_videos")]:
        ids = set(split.get(ids_key, []))
        subset = [v for v in all_videos if v["video_id"] in ids]

        loader = DataLoader(
            MultimodalVideoDataset(subset, cfg.MAX_SEQ_LEN),
            batch_size=1, shuffle=False
        )

        all_preds, all_tgts = [], []
        with torch.no_grad():
            for vid, batch in zip(subset, loader):
                v = batch["visual_features"]
                t = batch["text_features"]
                tgts = batch["importance_scores"].squeeze(-1).numpy()[0]
                slen = batch["seq_len"].item()
                preds = model(v, t).squeeze(-1).numpy()[0]
                all_preds.extend(preds[:slen])
                all_tgts.extend(tgts[:slen])

        metrics = compute_metrics(all_preds, all_tgts)
        results[split_name] = {
            "num_videos": len(subset),
            "num_segments": len(all_preds),
            "metrics": metrics
        }
        print(f"\n  {split_name}: {len(subset)} videos, {len(all_preds)} segments")
        print(f"  MSE: {metrics['mse']:.6f}  Pearson r: {metrics['pearson_r']:.4f}")
        print(f"  Pred std: {metrics['pred_std']:.4f}  Target std: {metrics['target_std']:.4f}")

    save_path = os.path.join(OUTPUT_DIR, "exp4_cross_split.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {save_path}")
    return results


def experiment_5_score_distribution_analysis(test_vids, models):
    """Experiment 5: Detailed analysis of predicted vs ground-truth score distributions."""
    print("\n" + "="*60)
    print("EXPERIMENT 5: Score Distribution Analysis")
    print("="*60)

    model = models["pth"]["model"]
    loader = DataLoader(
        MultimodalVideoDataset(test_vids, cfg.MAX_SEQ_LEN),
        batch_size=1, shuffle=False
    )

    all_preds, all_tgts = [], []
    with torch.no_grad():
        for vid, batch in zip(test_vids, loader):
            v = batch["visual_features"]
            t = batch["text_features"]
            tgts = batch["importance_scores"].squeeze(-1).numpy()[0]
            slen = batch["seq_len"].item()
            preds = model(v, t).squeeze(-1).numpy()[0]
            all_preds.extend(preds[:slen])
            all_tgts.extend(tgts[:slen])

    preds = np.array(all_preds)
    tgts = np.array(all_tgts)

    # Histogram bins
    bins = np.arange(0, 1.05, 0.05)
    pred_hist, _ = np.histogram(preds, bins=bins)
    tgt_hist, _ = np.histogram(tgts, bins=bins)

    # Percentiles
    percentiles = [5, 10, 25, 50, 75, 90, 95]

    results = {
        "predictions": {
            "histogram_bins": bins.tolist(),
            "histogram_counts": pred_hist.tolist(),
            "percentiles": {str(p): round(float(np.percentile(preds, p)), 4) for p in percentiles},
            "statistics": {
                "mean": round(float(preds.mean()), 4),
                "std": round(float(preds.std()), 4),
                "skewness": round(float(((preds - preds.mean()) ** 3).mean() / (preds.std() ** 3 + 1e-8)), 4),
                "kurtosis": round(float(((preds - preds.mean()) ** 4).mean() / (preds.std() ** 4 + 1e-8) - 3), 4),
            }
        },
        "ground_truth": {
            "histogram_bins": bins.tolist(),
            "histogram_counts": tgt_hist.tolist(),
            "percentiles": {str(p): round(float(np.percentile(tgts, p)), 4) for p in percentiles},
            "statistics": {
                "mean": round(float(tgts.mean()), 4),
                "std": round(float(tgts.std()), 4),
                "skewness": round(float(((tgts - tgts.mean()) ** 3).mean() / (tgts.std() ** 3 + 1e-8)), 4),
                "kurtosis": round(float(((tgts - tgts.mean()) ** 4).mean() / (tgts.std() ** 4 + 1e-8) - 3), 4),
            }
        },
        "residuals": {
            "mean": round(float((preds - tgts).mean()), 4),
            "std": round(float((preds - tgts).std()), 4),
            "max_abs": round(float(np.abs(preds - tgts).max()), 4),
        },
        "raw_predictions": preds.tolist(),
        "raw_targets": tgts.tolist(),
    }

    print(f"  Pred: mean={preds.mean():.4f}, std={preds.std():.4f}, range=[{preds.min():.4f}, {preds.max():.4f}]")
    print(f"  GT:   mean={tgts.mean():.4f}, std={tgts.std():.4f}, range=[{tgts.min():.4f}, {tgts.max():.4f}]")
    print(f"  Residual: mean={float((preds-tgts).mean()):.4f}, std={float((preds-tgts).std()):.4f}")

    save_path = os.path.join(OUTPUT_DIR, "exp5_score_distributions.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {save_path}")
    return results


def experiment_6_model_parameters(models):
    """Experiment 6: Model architecture analysis and parameter count."""
    print("\n" + "="*60)
    print("EXPERIMENT 6: Model Architecture Analysis")
    print("="*60)

    model = models["pth"]["model"]

    layer_info = []
    total_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        n = param.numel()
        total_params += n
        if param.requires_grad:
            trainable_params += n
        layer_info.append({
            "name": name,
            "shape": list(param.shape),
            "params": n,
            "trainable": param.requires_grad,
        })

    results = {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": round(total_params * 4 / (1024 * 1024), 2),  # float32
        "architecture": {
            "visual_projection": "Linear(512→128) + BN + ReLU + Dropout(0.2)",
            "text_projection": "Linear(384→128) + BN + ReLU + Dropout(0.2)",
            "fusion": "Concatenation(128+128=256) + Linear(256→256) + ReLU + Dropout(0.2)",
            "temporal_encoder": "BiLSTM(input=256, hidden=128, layers=2, bidirectional=True)",
            "classifier": "Linear(256→128) + ReLU + Dropout(0.2) + Linear(128→1) + Sigmoid",
        },
        "layer_details": layer_info,
    }

    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size (float32): {results['model_size_mb']:.2f} MB")

    save_path = os.path.join(OUTPUT_DIR, "exp6_model_architecture.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {save_path}")
    return results


def experiment_7_inference_timing(test_vids, models):
    """Experiment 7: Measure inference latency of importance model."""
    print("\n" + "="*60)
    print("EXPERIMENT 7: Inference Timing Analysis")
    print("="*60)

    model = models["pth"]["model"]
    loader = DataLoader(
        MultimodalVideoDataset(test_vids, cfg.MAX_SEQ_LEN),
        batch_size=1, shuffle=False
    )

    # Warm-up
    for vid, batch in zip(test_vids[:2], loader):
        with torch.no_grad():
            _ = model(batch["visual_features"], batch["text_features"])
        break

    timing_results = []
    loader = DataLoader(
        MultimodalVideoDataset(test_vids, cfg.MAX_SEQ_LEN),
        batch_size=1, shuffle=False
    )

    for vid, batch in zip(test_vids, loader):
        v = batch["visual_features"]
        t = batch["text_features"]
        slen = batch["seq_len"].item()

        # Time 10 runs
        times = []
        for _ in range(10):
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(v, t)
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)  # ms

        timing_results.append({
            "video_id": vid["video_id"],
            "seq_len": slen,
            "mean_ms": round(np.mean(times), 2),
            "std_ms": round(np.std(times), 2),
            "min_ms": round(np.min(times), 2),
            "max_ms": round(np.max(times), 2),
        })

    results = {
        "device": "cpu",
        "num_runs_per_video": 10,
        "per_video_timing": timing_results,
        "aggregate": {
            "mean_ms": round(np.mean([t["mean_ms"] for t in timing_results]), 2),
            "std_ms": round(np.std([t["mean_ms"] for t in timing_results]), 2),
            "min_ms": round(np.min([t["min_ms"] for t in timing_results]), 2),
            "max_ms": round(np.max([t["max_ms"] for t in timing_results]), 2),
        }
    }

    print(f"  Mean inference time: {results['aggregate']['mean_ms']:.2f} ms")
    print(f"  Range: [{results['aggregate']['min_ms']:.2f}, {results['aggregate']['max_ms']:.2f}] ms")

    save_path = os.path.join(OUTPUT_DIR, "exp7_inference_timing.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {save_path}")
    return results


def experiment_8_sensitivity_analysis(test_vids, models):
    """Experiment 8: Threshold sensitivity analysis for binary classification."""
    print("\n" + "="*60)
    print("EXPERIMENT 8: Threshold Sensitivity Analysis")
    print("="*60)

    model = models["pth"]["model"]
    loader = DataLoader(
        MultimodalVideoDataset(test_vids, cfg.MAX_SEQ_LEN),
        batch_size=1, shuffle=False
    )

    all_preds, all_tgts = [], []
    with torch.no_grad():
        for vid, batch in zip(test_vids, loader):
            v = batch["visual_features"]
            t = batch["text_features"]
            tgts = batch["importance_scores"].squeeze(-1).numpy()[0]
            slen = batch["seq_len"].item()
            preds = model(v, t).squeeze(-1).numpy()[0]
            all_preds.extend(preds[:slen])
            all_tgts.extend(tgts[:slen])

    preds = np.array(all_preds)
    tgts = np.array(all_tgts)

    thresholds = np.arange(0.1, 1.0, 0.05)
    threshold_results = []

    for thresh in thresholds:
        bp = (preds >= thresh).astype(int)
        bt = (tgts >= thresh).astype(int)
        tp = int(np.sum((bp == 1) & (bt == 1)))
        fp = int(np.sum((bp == 1) & (bt == 0)))
        fn = int(np.sum((bp == 0) & (bt == 1)))
        tn = int(np.sum((bp == 0) & (bt == 0)))
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        threshold_results.append({
            "threshold": round(float(thresh), 2),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })

    results = {"threshold_analysis": threshold_results}

    # Find optimal threshold
    best = max(threshold_results, key=lambda x: x["f1_score"])
    results["optimal_threshold"] = best["threshold"]
    results["optimal_f1"] = best["f1_score"]

    print(f"  Optimal threshold: {best['threshold']:.2f} (F1={best['f1_score']:.4f})")
    for tr in threshold_results:
        print(f"    thresh={tr['threshold']:.2f}: P={tr['precision']:.4f} R={tr['recall']:.4f} F1={tr['f1_score']:.4f}")

    save_path = os.path.join(OUTPUT_DIR, "exp8_threshold_sensitivity.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {save_path}")
    return results


def main():
    print("=" * 60)
    print("RESEARCH EVALUATION — EXPERIMENTAL SUITE")
    print("=" * 60)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output directory: {OUTPUT_DIR}")

    start_time = time.time()

    # Load data
    all_videos, train_vids, val_vids, test_vids, models, split = load_data_and_model()
    print(f"\nData loaded: {len(all_videos)} total, {len(train_vids)} train, {len(val_vids)} val, {len(test_vids)} test")
    print(f"Models loaded: {list(models.keys())}")

    # Run experiments
    all_results = {}

    all_results["exp1"] = experiment_1_model_evaluation(test_vids, models)
    all_results["exp2"] = experiment_2_per_dataset_evaluation(test_vids, models)
    all_results["exp3"] = experiment_3_ablation_study(test_vids, models)
    all_results["exp4"] = experiment_4_cross_split_evaluation(all_videos, models, split)
    all_results["exp5"] = experiment_5_score_distribution_analysis(test_vids, models)
    all_results["exp6"] = experiment_6_model_parameters(models)
    all_results["exp7"] = experiment_7_inference_timing(test_vids, models)
    all_results["exp8"] = experiment_8_sensitivity_analysis(test_vids, models)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"ALL EXPERIMENTS COMPLETE in {elapsed:.1f} seconds")
    print(f"Results saved in: {OUTPUT_DIR}")
    print(f"{'='*60}")

    # Save master results
    master_path = os.path.join(OUTPUT_DIR, "master_results.json")
    with open(master_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_time_seconds": round(elapsed, 1),
            "experiments_run": list(all_results.keys()),
        }, f, indent=2)

    return all_results


if __name__ == "__main__":
    main()
