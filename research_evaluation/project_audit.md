# Project Audit Report
## A Multimodal Framework for Long-Format Video Summarization Using Visual Importance Estimation and Large Language Models

**Audit Date:** 2026-10-29  
**Auditor:** Automated Code Inspection  
**Project Root:** `c:\Users\ravindra_007\OneDrive\Desktop\ai prioritizer\video-summarizer`

---

## 1. Executive Summary

The project implements a video summarization pipeline with a React frontend and Python FastAPI backend. A custom BiLSTM-based multimodal importance model scores temporal segments using fused visual and text features, and a T5-small model generates abstractive text summaries from selected transcript chunks.

**Critical finding:** The actual implementation differs substantially from the intended methodology described in the research proposal. None of the originally proposed components (VASNet, GoogLeNet/Inception-v1, KTS, Qwen2.5-0.5B-Instruct, SmolVLM-256M-Instruct, knapsack algorithm) are implemented. The model was trained on **synthetic** benchmark data, not on real SumMe/TVSum videos or LVSum. The "multimodal" claim applies to the importance scoring stage only—the final text summary generation uses transcript text exclusively.

---

## 2. System Environment

| Property | Value |
|----------|-------|
| OS | Windows 11 Home Single Language |
| CPU | AMD Ryzen 5 5600H (6 cores, 12 threads) |
| RAM | ~16 GB |
| GPU | AMD Radeon (integrated) — no CUDA |
| Python | 3.10.11 |
| PyTorch | 2.11.0 (CPU only) |
| Transformers | 5.17.0 |
| faster-whisper | 1.2.1 |
| FastAPI | 0.141.1 |
| OpenCV | 5.0.0.93 |
| sentence-transformers | 6.0.1 |

---

## 3. Technology Stack

### Frontend
- **Framework:** React 18 + Vite
- **Purpose:** Video upload UI, YouTube URL processing, progress tracking, summary display

### Backend
- **Framework:** Python FastAPI + Uvicorn
- **API Endpoints:**
  - `POST /api/upload` — File upload
  - `POST /api/process-url` — YouTube URL processing (via yt-dlp)
  - `POST /api/process/{task_id}` — Trigger processing
  - `GET /api/status/{task_id}` — Progress polling
  - `GET /api/result/{task_id}` — Retrieve results
  - `POST /api/video-info` — Quick YouTube metadata

---

## 4. Actual Architecture (Verified from Source Code)

### 4.1 Pipeline Flow (backend/services/pipeline.py)

```
Video Input (upload or YouTube URL download)
    │
    ├─► VideoProcessor.extract_metadata()        → duration, fps, resolution
    │
    ├─► AudioProcessor.process_audio()
    │     ├─ FFmpeg audio extraction → WAV
    │     └─ WhisperService.transcribe()          → timestamped transcript segments
    │
    ├─► FrameProcessor.process_video_frames()
    │     ├─ OpenCV frame sampling (1 frame / 15 seconds)
    │     └─ ResNet-18 feature extraction         → 512-dim per frame
    │
    ├─► TemporalAligner.align_and_chunk()         → 180-second chunks
    │     ├─ Transcript text assigned per chunk
    │     └─ Visual features mean-pooled per chunk
    │
    ├─► VideoImportanceService.score_chunks()
    │     ├─ SentenceTransformer text encoding     → 384-dim per chunk
    │     ├─ BiLSTM importance model inference     → score ∈ [0,1]
    │     └─ Coverage-aware temporal selection
    │
    └─► HierarchicalSummarizer
          ├─ T5-small per-chunk summarization (transcript only)
          └─ T5-small final summary aggregation (transcript only)
```

### 4.2 Visual Feature Extraction (backend/services/visual_features.py)

| Property | Actual Value | Intended Value |
|----------|-------------|----------------|
| CNN Backbone | **ResNet-18** (pretrained, ImageNet) | GoogLeNet/Inception-v1 |
| Feature Dimension | **512** | 1024 |
| Input Size | 224×224 | - |
| Normalization | ImageNet mean/std | - |
| Framework | torchvision.models | - |

The final FC layer is replaced with `nn.Identity()` to extract penultimate-layer embeddings.

**`generate_visual_description()`** is a stub that always returns a hardcoded string: `"Visual frame showing lecture slide or video scene activity."` — visual captions are not actually generated.

### 4.3 Frame Sampling (backend/services/frame_processor.py)

| Property | Actual Value | Intended Value |
|----------|-------------|----------------|
| Sampling Rate | **1 frame every 15 seconds** (~0.067 fps) | 2 fps |
| Method | OpenCV `CAP_PROP_POS_FRAMES` seek | - |

**Note:** The training config specifies `FRAME_SAMPLE_FPS = 2` for feature extraction from SumMe/TVSum, but the production inference pipeline uses `frame_interval_seconds = 15` from `configs/config.yaml`. This is a significant discrepancy—the model was trained on 2fps features but at inference time receives features sampled at ~0.067fps.

### 4.4 Speech Transcription (backend/services/whisper_service.py)

| Property | Actual Value | Intended Value |
|----------|-------------|----------------|
| Model | **Faster-Whisper (tiny)** | Faster-Whisper |
| Device | CPU | - |
| Compute Type | int8 | - |
| Language | English (forced) | - |
| Beam Size | 1 | - |
| VAD Filter | Enabled (min_silence=500ms) | - |

Falls back to synthetic placeholder segments if transcription fails.

### 4.5 Importance Model (training/model.py)

| Property | Actual Value | Intended Value |
|----------|-------------|----------------|
| Architecture | **Custom BiLSTM Fusion Model** | VASNet (attention-based) |
| Visual Input | 512-dim (ResNet-18) | 1024-dim (GoogLeNet) |
| Text Input | 384-dim (all-MiniLM-L6-v2) | Not in VASNet |
| Visual Projection | Linear(512→128) + BN + ReLU + Dropout | - |
| Text Projection | Linear(384→128) + BN + ReLU + Dropout | - |
| Fusion | Concatenation (128+128=256) + MLP | - |
| Temporal Encoder | BiLSTM (2 layers, hidden=128, bidirectional) | Self-attention (VASNet) |
| Classifier | Linear(256→128→1) + Sigmoid | - |
| Total Parameters | **1,005,315** | - |
| Output | Importance score ∈ [0,1] per segment | - |

### 4.6 Temporal Segmentation (backend/services/temporal_alignment.py)

| Property | Actual Value | Intended Value |
|----------|-------------|----------------|
| Method | **Fixed-duration chunking** (180 seconds) | KTS (Kernel Temporal Segmentation) |
| Visual Aggregation | Mean-pooling of frame features within chunk | - |
| Text Aggregation | Concatenation of transcript segments within chunk | - |

No adaptive segmentation is performed. Chunks are uniform 3-minute windows.

### 4.7 Segment Selection (backend/services/importance_model.py)

| Property | Actual Value | Intended Value |
|----------|-------------|----------------|
| Method | **Coverage-aware temporal windowing** | Knapsack algorithm |
| Logic | Divide timeline into N equal windows, pick highest-scoring chunk per window | - |
| Target Count | For ≤15 chunks: select all; For >15: ~85% | Summary-length budget |

**Score-scaling hack (lines 121-131):** If the model's score range is < 0.15 (near-flat predictions), scores are artificially spread between 0.42–0.96 using heuristic rules combining transcript length and visual feature variance.

### 4.8 Text Summarization (backend/services/summarizer.py)

| Property | Actual Value | Intended Value |
|----------|-------------|----------------|
| Model | **T5-small** (60M params, pretrained) | Qwen2.5-0.5B-Instruct |
| Fallback | OpenAI GPT-3.5-turbo API (if key set) | - |
| Visual Captioning | **Not implemented** (stub returns hardcoded string) | SmolVLM-256M-Instruct |
| Input to LLM | **Transcript text only** | Multimodal (visual + text) |

**Critical:** The text summarizer receives ONLY transcript text. Visual features are NOT passed to the LLM. The "multimodal" aspect operates exclusively at the importance scoring stage, not at the summary generation stage.

### 4.9 Config (configs/config.yaml)

| Setting | Value |
|---------|-------|
| `system.chunk_duration_seconds` | 180 |
| `system.frame_interval_seconds` | 15 |
| `audio.whisper_model` | tiny |
| `visual.extractor` | resnet18 |
| `model.importance_threshold` | 0.5 |
| `model.top_k_chunks` | 15 |
| `llm.provider` | mock_api |
| `llm.api_key` | (empty) |
| `paths.model_save_path` | models/video_importance_model.pt |

---

## 5. Training Data Assessment

### 5.1 Dataset Source

The training used **synthetic benchmark data** generated by `training/preprocess.py`:
- 25 synthetic "SumMe" videos with random 512-dim visual features and 384-dim text features
- 50 synthetic "TVSum" videos with the same structure
- Importance scores derived from random feature statistics with temporal bias
- **No real video frames or transcripts were used for training**

### 5.2 Evidence of Synthetic Data

1. `TRAINING.md` line 181: *"Training on **synthetic** benchmark data (75 videos: 25 SumMe + 50 TVSum)."*
2. `preprocess.py` generates random Gaussian features normalized to unit norm
3. Scores are constructed from: `0.3 + 0.35 * visual_std_signal + 0.30 * text_signal + temporal_bias + noise`
4. Video IDs follow pattern `summe_video_XX` and `tvsum_video_XX` (not real video names)
5. `feature_extraction.py` exists but requires Kaggle API credentials and real dataset download

### 5.3 Feature Extraction Pipeline (training/feature_extraction.py)

A real feature extraction pipeline exists that would:
1. Download SumMe/TVSum from Kaggle via kagglehub
2. Sample frames at 2 fps
3. Extract ResNet-18 features
4. Load ground-truth annotations (HDF5 for SumMe, TSV for TVSum)
5. Align features with annotations

**Status:** This pipeline was NEVER executed (no `.npy` cache files exist in `datasets/video_features/`).

### 5.4 Data Split

| Split | Count | Ratio |
|-------|-------|-------|
| Train | 52 videos | 70% |
| Validation | 11 videos | 15% |
| Test | 12 videos | 15% |

Split is video-level (no data leakage). Saved in `datasets/data_split.json`.

---

## 6. Model Checkpoints

### 6.1 `models/video_importance_model.pt` (Production)
- Epoch: 8, Val Loss: 0.001107, ~3.85 MB
- Config: visual_dim=512, text_dim=384, fused_dim=256, hidden_dim=128

### 6.2 `models/video_importance_model.pth` (Latest training)
- Epoch: 18, Val Loss: 0.000447, ~3.85 MB  
- Same architecture config

Both trained on synthetic data. The `.pt` file (epoch 8) is the one loaded by the production pipeline.

---

## 7. Existing Evaluation Results (evaluation/results.json)

These results are from `training/evaluate.py` on the `.pth` checkpoint (epoch 18):

| Metric | Value |
|--------|-------|
| MSE | 0.007335 |
| MAE | 0.050162 |
| Pearson r | **-0.0506** |
| Pearson p-value | 0.4264 |
| Precision (at 0.5) | 1.0000 |
| Recall (at 0.5) | 0.9880 |
| F1 Score (at 0.5) | 0.9939 |
| Pred Variance | 0.005072 |

**Interpretation:**
- The near-zero Pearson r (-0.05) indicates **no meaningful linear correlation** between predicted and ground-truth importance scores.
- The artificially high precision/recall/F1 is because synthetic ground-truth scores are mostly above 0.5 (range ~0.76–0.95) and the model also predicts mostly above 0.5 — the binary classification threshold is trivially satisfied.
- The low MSE (0.007) is misleading because both predictions and targets cluster in a narrow range around 0.85.
- Pred variance of 0.005 confirms predictions have very low dynamic range.

---

## 8. Discrepancy Summary: Intended vs. Actual

| Component | Intended | Actual | Impact |
|-----------|----------|--------|--------|
| Visual CNN | GoogLeNet (1024-dim) | ResNet-18 (512-dim) | Different feature representation |
| Importance Model | VASNet (attention) | BiLSTM (recurrent) | Different temporal modeling paradigm |
| Temporal Segmentation | KTS (adaptive) | Fixed 180s chunks | No content-adaptive boundaries |
| Segment Selection | Knapsack (budget-constrained) | Coverage-aware windowing | Different optimization objective |
| Text Summarizer | Qwen2.5-0.5B-Instruct | T5-small (60M) | Much smaller model, different architecture |
| Visual Captioning | SmolVLM-256M-Instruct | Not implemented (stub) | No visual descriptions in output |
| Training Data | LVSum dataset | Synthetic random features | No real visual/text patterns learned |
| Frame Sampling (training) | 2 fps | 2 fps (synthetic) | N/A — synthetic data |
| Frame Sampling (inference) | 2 fps | 1 frame / 15s (~0.067fps) | 30× fewer frames than intended |
| Multimodal Summary | Visual + text fusion at output | Transcript-only at output | Not multimodal at summary generation |

---

## 9. Strengths

1. Clean, modular code architecture with clear separation of concerns
2. Proper video-level data split prevents leakage
3. Caching system for transcripts, frames, and visual features
4. Progress reporting through the pipeline
5. Fallback mechanisms at multiple stages
6. Working end-to-end pipeline from video upload to summary generation
7. YouTube URL support via yt-dlp

---

## 10. Weaknesses and Concerns

1. **Model trained on synthetic data** — no real visual patterns or annotation signals learned
2. **Near-zero Pearson correlation** on test set — model does not meaningfully rank segment importance
3. **Score-scaling hack** in production masks poor model discrimination
4. **No multimodal fusion at summary generation** — LLM only sees transcript text
5. **Visual captioning is a stub** — returns hardcoded string
6. **Train-inference feature mismatch** — 2fps training vs 15s-interval inference
7. **T5-small has limited abstractive capability** for long-form summarization
8. **No KTS or adaptive segmentation** — fixed chunks may split semantically coherent segments

---

## 11. Files Inventory

### Backend Services
- `backend/main.py` — FastAPI app entry point
- `backend/api/routes.py` — API endpoint definitions (287 lines)
- `backend/services/pipeline.py` — Main orchestration (99 lines)
- `backend/services/video_processor.py` — Video metadata extraction (39 lines)
- `backend/services/audio_processor.py` — Audio extraction + Whisper (36 lines)
- `backend/services/whisper_service.py` — Faster-Whisper transcription (93 lines)
- `backend/services/frame_processor.py` — Frame sampling + visual features (88 lines)
- `backend/services/visual_features.py` — ResNet-18 feature extractor (44 lines)
- `backend/services/temporal_alignment.py` — Fixed-duration chunking (50 lines)
- `backend/services/importance_model.py` — BiLSTM inference + selection (154 lines)
- `backend/services/summarizer.py` — T5-small hierarchical summarizer (233 lines)
- `backend/utils/config_loader.py` — YAML config loader (18 lines)

### Training
- `training/model.py` — MultimodalImportanceModel (89 lines)
- `training/train.py` — Training loop with early stopping (195 lines)
- `training/evaluate.py` — Evaluation script (176 lines)
- `training/dataset.py` — PyTorch Dataset class (49 lines)
- `training/config.py` — Training hyperparameters (51 lines)
- `training/preprocess.py` — Synthetic data generation (118 lines)
- `training/feature_extraction.py` — Real data pipeline (389 lines, never executed)
- `training/download_dataset.py` — Kaggle download (7 lines)

### Evaluation
- `evaluation/evaluate_importance.py` — Evaluation metrics (109 lines)
- `evaluation/results.json` — Existing results (272 lines)
