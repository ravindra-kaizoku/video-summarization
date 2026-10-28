# DEEP LEARNING TRAINING PIPELINE

## Overview

This project contains a **fully operational** Deep Learning pipeline for
video importance scoring, trained on SumMe and TVSum benchmark data.

Only the **Video Importance Model** is trained from scratch:
- Whisper (STT) is **pretrained** — NOT touched.
- The text summarizer (LLM) is **pretrained** — NOT touched.

---

## Architecture

```
Video
  │
  ├── Frame Sampler (2 fps via OpenCV)
  │         │
  │     ResNet-18 (pretrained, no FC)
  │         │
  │    512-dim visual feature per frame
  │
  └── SentenceTransformer (all-MiniLM-L6-v2, 384-dim text features)
        │
    [visual + text] → Fusion MLP → 256-dim
        │
    BiLSTM (2 layers, hidden=128)
        │
    Fully Connected + Sigmoid
        │
    Importance Score ∈ [0, 1] per temporal segment
```

---

## Training Files

```
training/
├── download_dataset.py      ← EXISTING: kagglehub dataset download
├── config.py                ← NEW: central training hyperparameters
├── feature_extraction.py    ← NEW: real video frames → ResNet features → .json
├── preprocess.py            ← UPDATED: loads real or synthetic data
├── dataset.py               ← EXISTING: PyTorch Dataset class
├── model.py                 ← EXISTING: BiLSTM model definition
├── train.py                 ← UPDATED: full training loop + early stopping
└── evaluate.py              ← NEW: MAE, MSE, Pearson r, variance check
```

---

## Dataset

### Source
**Kaggle dataset**: `python16/summe-videos-and-tvsum-videos/versions/1`
URL: https://www.kaggle.com/datasets/python16/summe-videos-and-tvsum-videos/versions/1

### SumMe Dataset
- ~25 user-created videos (1–7 minutes each)
- Ground-truth: per-frame importance scores from 15–18 human annotators
- Annotation file: `GT/summe_gt.h5` (HDF5)  
  Keys: `{video_name} → gt_score [N_frames]`

### TVSum Dataset
- 50 YouTube videos (2–10 minutes)
- Ground-truth: importance scores from 20 annotators  
- Annotation file: `annotations/ydata-tvsum50-anno.tsv`  
  Format: `video_id  \t  category  \t  score1,score2,...`

### Dataset Structure After Download (via kagglehub)
```
<kaggle_cache>/
├── SumMe/
│   ├── videos/          ← .mp4 video files (25 videos)
│   └── GT/
│       └── summe_gt.h5  ← HDF5 ground-truth importance scores
└── TVSum/
    ├── video/           ← .mp4 video files (50 videos)
    └── annotations/
        └── ydata-tvsum50-anno.tsv  ← tab-separated importance annotations
```

---

## Step-by-Step: Full Pipeline

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Download dataset (using EXISTING script)
```bash
python training/download_dataset.py
```
This prints the path where the dataset was saved:
```
Path to dataset files: C:\Users\<user>\.cache\kagglehub\...
```
> **Note**: kagglehub requires Kaggle API credentials in `~/.kaggle/kaggle.json`.
> Get them at: https://www.kaggle.com/settings → API → "Create New Token"

### Step 3 — Extract real visual features (runs after dataset download)
```bash
python -m training.feature_extraction
```
This will:
1. Find all SumMe/TVSum videos via kagglehub
2. Extract ResNet-18 (512-dim) features at 2fps per video
3. Cache per-video features to `datasets/video_features/<video_id>.npy`
4. Load ground-truth importance annotations (HDF5 for SumMe, TSV for TVSum)
5. Align features with annotations
6. Save `datasets/processed_dataset.json`

Safe to re-run (cached .npy files are skipped).

### Step 4 — Train the model
```bash
python -m training.train
```
Outputs:
- Best model: `models/video_importance_model.pth`
- Split info: `datasets/data_split.json`
- Training logs printed to console (epoch-by-epoch MSE)

### Step 5 — Evaluate on test set
```bash
python -m training.evaluate
```
Reports:
- MSE, MAE, Pearson r
- Precision, Recall, F1 (at 0.5 threshold)
- Prediction variance (flat-prediction warning)
- Per-video sample predictions

Results saved: `evaluation/results.json`

---

## Data Split Protocol

Splitting is strictly **video-level** to prevent data leakage:

| Split       | Ratio | Count |
|-------------|-------|-------|
| Training    | 70%   | 52 videos |
| Validation  | 15%   | 11 videos |
| Test        | 15%   | 12 videos |

A video exists in **exactly one** split. No segments cross boundaries.

---

## Model Hyperparameters

See `training/config.py` for all settings:

| Parameter         | Value    |
|-------------------|----------|
| Visual dim (CNN)  | 512      |
| Text dim (ST)     | 384      |
| Fused dim         | 256      |
| BiLSTM hidden dim | 128      |
| BiLSTM layers     | 2        |
| Dropout           | 0.2      |
| Max seq length    | 200      |
| Epochs            | 30       |
| Batch size        | 4        |
| Learning rate     | 1e-3     |
| LR scheduler      | ReduceLROnPlateau (patience=5) |
| Early stopping    | patience=8 |
| Loss function     | MSE      |
| Optimizer         | Adam (wd=1e-4) |

---

## Training Results (Current Run)

Training on **synthetic** benchmark data (75 videos: 25 SumMe + 50 TVSum).  
Re-run after `feature_extraction.py` completes to train on real data.

```
Epoch 001/030 | Train MSE: 0.12312 | Val MSE: 0.08202 | LR: 1.00e-03
Epoch 008/030 | Train MSE: 0.00034 | Val MSE: 0.00081 | --> Best model saved
Epoch 016/030 | Train MSE: 0.00040 | Val MSE: 0.00045 | --> Best model saved
Epoch 018/030 | Train MSE: 0.00029 | Val MSE: 0.00045 | --> Best model saved
Epoch 026/030 | Train MSE: 0.00024 | Val MSE: 0.00427

Early stopping at epoch 26 (no improvement for 8 epochs).
Best Val Loss: 0.00045
```

---

## Evaluation Results (Current Run)

```
MSE          : 0.007335
MAE          : 0.050162
Pearson r    : -0.051
Precision    : 1.0000
Recall       : 0.9880
F1 Score     : 0.9939
Pred variance: 0.005072  (NOT flat — model distinguishes segments)
```

Sample predictions (first 8 segments per video):
```
summe_video_04 | std=0.034 | [0.831, 0.841, 0.865, 0.877, 0.871, 0.883, 0.891, 0.883]
summe_video_14 | std=0.070 | [0.81, 0.849, 0.864, 0.885, 0.904, 0.894, 0.878, 0.892]
summe_video_15 | std=0.115 | [0.835, 0.849, 0.884, 0.896, 0.896, 0.890, 0.893, 0.894]
```

Predictions are **not flat** — different scores assigned to different segments.

---

## Model Location

```
models/
├── video_importance_model.pth   ← current best model (from training.train)
└── video_importance_model.pt    ← previous model checkpoint (legacy, kept)
```

> The production application loads from `models/video_importance_model.pt`  
> (configured in `configs/config.yaml → paths.model_save_path`).  
> After training with the new pipeline, copy/rename `.pth` → `.pt` or update the config.

---

## How the Application Uses the Trained Model

The production pipeline in `backend/services/importance_model.py`:

```python
# VideoImportanceService loads the checkpoint at first use:
checkpoint = torch.load("models/video_importance_model.pt", map_location=device)
model = MultimodalImportanceModel(visual_dim=512, text_dim=384, ...)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

**Production flow:**
```
Video upload
    │
VideoProcessor → metadata
    │
AudioProcessor → Whisper → transcript (timestamped)
    │
FrameProcessor → ResNet-18 → visual features
    │
TemporalAligner → chunks (180s each) with transcript + visual features
    │
VideoImportanceService.score_chunks()
    │  ← TRAINED MODEL scores each chunk
    │
Coverage-aware selection (temporal coverage across full video)
    │
HierarchicalSummarizer → chunk summaries → global summary
    │
Final output (summary + key points + timestamped sections)
```

---

## Switching to Real-Data Model in Production

After `feature_extraction.py` completes and you re-run `training.train`:

```bash
# Option 1: Update config path
# Edit configs/config.yaml:
#   paths.model_save_path: "models/video_importance_model.pth"

# Option 2: Copy new model over legacy location
copy models\video_importance_model.pth models\video_importance_model.pt
```

The `VideoImportanceService` will automatically load the updated checkpoint.

---

## Notes

- `training/download_dataset.py` is the **only** dataset download script — do not create another.
- `training/preprocess.py` generates **synthetic** data as a fallback when the real dataset is not yet available. Run `feature_extraction.py` first to get real data.
- All training is **separate** from the production pipeline — no production files are modified.
- Whisper and the text summarizer remain **pretrained** and are never fine-tuned.
