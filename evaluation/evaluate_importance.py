import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from training.model import MultimodalImportanceModel
from training.dataset import MultimodalVideoDataset

def evaluate_importance_model(
    model_path: str = "models/video_importance_model.pt",
    dataset_path: str = "datasets/processed_dataset.json",
    split_path: str = "datasets/data_split.json",
    output_json_path: str = "evaluation/results.json"
):
    print("=== EVALUATING VIDEO IMPORTANCE MODEL ===")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model checkpoint not found at: {model_path}")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split file not found at: {split_path}")

    # Load data split
    with open(split_path, "r", encoding="utf-8") as f:
        split_info = json.load(f)
    test_video_ids = set(split_info.get("test_videos", []))

    # Load dataset
    with open(dataset_path, "r", encoding="utf-8") as f:
        all_videos = json.load(f)

    test_videos = [v for v in all_videos if v["video_id"] in test_video_ids]
    if not test_videos:
        test_videos = all_videos[-10:]  # Fallback to last 10 videos if split not matched

    test_dataset = MultimodalVideoDataset(test_videos)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Load trained model
    checkpoint = torch.load(model_path, map_location="cpu")
    model = MultimodalImportanceModel(visual_dim=512, text_dim=384, fused_dim=256, hidden_dim=128)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            v_feats = batch["visual_features"]
            t_feats = batch["text_features"]
            targets = batch["importance_scores"].squeeze(-1).numpy()[0]
            seq_len = batch["seq_len"].item()

            preds = model(v_feats, t_feats).squeeze(-1).numpy()[0]

            valid_preds = preds[:seq_len]
            valid_targets = targets[:seq_len]

            all_preds.extend(valid_preds)
            all_targets.extend(valid_targets)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Calculate MSE
    mse = float(np.mean((all_preds - all_targets) ** 2))

    # Binary Classification Metrics (Threshold at 0.5 importance)
    threshold = 0.5
    binary_preds = (all_preds >= threshold).astype(int)
    binary_targets = (all_targets >= threshold).astype(int)

    tp = np.sum((binary_preds == 1) & (binary_targets == 1))
    fp = np.sum((binary_preds == 1) & (binary_targets == 0))
    fn = np.sum((binary_preds == 0) & (binary_targets == 1))

    precision = float(tp / (tp + fp + 1e-8))
    recall = float(tp / (tp + fn + 1e-8))
    f1_score = float(2 * precision * recall / (precision + recall + 1e-8))

    results = {
        "model_path": model_path,
        "test_videos_count": len(test_videos),
        "total_test_segments": len(all_preds),
        "metrics": {
            "mse_loss": round(mse, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1_score, 4)
        }
    }

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Evaluation Results Saved to '{output_json_path}':")
    print(f" - MSE Loss: {mse:.4f}")
    print(f" - Precision: {precision:.4f}")
    print(f" - Recall: {recall:.4f}")
    print(f" - F1-Score: {f1_score:.4f}")

    return results

if __name__ == "__main__":
    evaluate_importance_model()
