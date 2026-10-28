"""
training/train.py
=================
Full PyTorch training pipeline for the MultimodalImportanceModel.

Usage (from project root):
    python -m training.train

Features:
  - Video-level data split (no data leakage)
  - MSE loss against human importance annotations
  - ReduceLROnPlateau learning-rate scheduler
  - Early stopping
  - Best-model checkpoint saving
  - Full training logs
"""

import os
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from training.model   import MultimodalImportanceModel
from training.dataset import MultimodalVideoDataset
from training.preprocess import load_or_generate_dataset
import training.config as cfg


def _set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(
    dataset_path: str  = None,
    model_save_path: str = None,
    epochs: int = None,
    batch_size: int = None,
    learning_rate: float = None,
    device: str = None,
):
    _set_seed(cfg.SEED)

    dataset_path    = dataset_path    or cfg.PROCESSED_DATASET_PATH
    model_save_path = model_save_path or cfg.MODEL_SAVE_PATH
    epochs          = epochs          or cfg.EPOCHS
    batch_size      = batch_size      or cfg.BATCH_SIZE
    learning_rate   = learning_rate   or cfg.LEARNING_RATE

    # Auto-detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else cfg.DEVICE
    print(f"=== VIDEO IMPORTANCE MODEL TRAINING ===")
    print(f"Device    : {device}")
    print(f"Epochs    : {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"LR        : {learning_rate}")

    # Ensure dataset exists
    if not os.path.exists(dataset_path):
        print("[Train] Dataset not found — generating...")
        load_or_generate_dataset(dataset_path)

    with open(dataset_path, "r", encoding="utf-8") as f:
        all_videos = json.load(f)

    # ── VIDEO-LEVEL SPLIT (prevents data leakage) ──────────────────────────
    random.shuffle(all_videos)
    n          = len(all_videos)
    train_end  = int(n * cfg.TRAIN_RATIO)
    val_end    = int(n * (cfg.TRAIN_RATIO + cfg.VAL_RATIO))

    train_vids = all_videos[:train_end]
    val_vids   = all_videos[train_end:val_end]
    test_vids  = all_videos[val_end:]

    print(f"\nData split (video-level, no leakage):")
    print(f"  Train: {len(train_vids)} | Val: {len(val_vids)} | Test: {len(test_vids)}")

    max_seq = cfg.MAX_SEQ_LEN
    train_loader = DataLoader(MultimodalVideoDataset(train_vids, max_seq), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(MultimodalVideoDataset(val_vids,   max_seq), batch_size=batch_size, shuffle=False)

    # ── MODEL ───────────────────────────────────────────────────────────────
    model = MultimodalImportanceModel(
        visual_dim=cfg.VISUAL_DIM,
        text_dim=cfg.TEXT_DIM,
        fused_dim=cfg.FUSED_DIM,
        hidden_dim=cfg.HIDDEN_DIM,
        num_layers=cfg.NUM_LAYERS,
        dropout=cfg.DROPOUT,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=cfg.SCHEDULER_PATIENCE,
                                   factor=0.5)

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)

    best_val_loss   = float("inf")
    no_improve_cnt  = 0
    training_log    = []

    # ── TRAINING LOOP ───────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        # --- Train phase ---
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            v = batch["visual_features"].to(device)
            t = batch["text_features"].to(device)
            y = batch["importance_scores"].to(device)

            optimizer.zero_grad()
            preds = model(v, t)
            loss  = criterion(preds, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * v.size(0)
        train_loss /= max(1, len(train_loader.dataset))

        # --- Validation phase ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                v = batch["visual_features"].to(device)
                t = batch["text_features"].to(device)
                y = batch["importance_scores"].to(device)
                preds    = model(v, t)
                val_loss += criterion(preds, y).item() * v.size(0)
        val_loss /= max(1, len(val_loader.dataset))

        scheduler.step(val_loss)

        if epoch % cfg.LOG_EVERY_N_EPOCHS == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:03d}/{epochs:03d} | "
                  f"Train MSE: {train_loss:.5f} | "
                  f"Val MSE: {val_loss:.5f} | LR: {lr_now:.2e}")

        training_log.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        # --- Best model checkpoint ---
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            no_improve_cnt = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch"           : epoch,
                "val_loss"        : val_loss,
                "config": {
                    "visual_dim": cfg.VISUAL_DIM,
                    "text_dim"  : cfg.TEXT_DIM,
                    "fused_dim" : cfg.FUSED_DIM,
                    "hidden_dim": cfg.HIDDEN_DIM,
                },
            }, model_save_path)
            print(f"  --> Best model saved (val_loss={val_loss:.5f})")
        else:
            no_improve_cnt += 1

        # --- Early stopping ---
        if no_improve_cnt >= cfg.EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {cfg.EARLY_STOP_PATIENCE} epochs).")
            break

    # ── SAVE SPLIT METADATA ─────────────────────────────────────────────────
    split_info = {
        "train_videos": [v["video_id"] for v in train_vids],
        "val_videos"  : [v["video_id"] for v in val_vids],
        "test_videos" : [v["video_id"] for v in test_vids],
    }
    with open(cfg.DATA_SPLIT_PATH, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    print(f"\nTraining complete. Best Val Loss: {best_val_loss:.5f}")
    print(f"Model saved to  : {model_save_path}")
    print(f"Split saved to  : {cfg.DATA_SPLIT_PATH}")
    return model_save_path


if __name__ == "__main__":
    train_model()
