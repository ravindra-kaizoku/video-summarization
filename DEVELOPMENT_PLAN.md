# DEVELOPMENT PLAN: VIDEO SUMMARIZATION USING LLMs

## System Environment Assessment
- **OS**: Windows 11 (AMD64)
- **Python**: 3.10.11
- **Node**: v25.2.1 | **npm**: 11.6.2
- **Device**: CPU (Torch 2.11.0+cpu) - Optimized for student CPU execution
- **Workspace Path**: `c:\Users\ravindra_007\OneDrive\Desktop\ai prioritizer\video-summarizer`

---

## 📌 Implementation Phases

### Phase 1: Project Structure & Environment Setup
- Create directory structure (`backend`, `training`, `datasets`, `models`, `frontend`, `evaluation`, `configs`, `scripts`).
- Create `requirements.txt` with FastAPI, Uvicorn, PyTorch, torchvision, faster-whisper/whisper, sentence-transformers, transformers, opencv-python, imageio-ffmpeg, pyyaml, pydantic.
- Create `configs/config.yaml` with long-video settings (`MIN_VIDEO_DURATION_MINUTES=30`, `CHUNK_DURATION_SECONDS=300`, `FRAME_INTERVAL_SECONDS=15`).

### Phase 2: Video Ingestion & Metadata Extraction
- Implement `video_processor.py` for duration detection, resolution, frame rate, audio stream presence, and multi-track handling using OpenCV and imageio-ffmpeg.

### Phase 3: Audio Extraction & Whisper Transcription
- Implement `audio_processor.py` for FFmpeg WAV extraction.
- Implement `whisper_service.py` with fallback mechanisms (Faster-Whisper / OpenAI Whisper / Mock fallback for offline/low-resource environments).
- Structure transcripts into timestamped JSON objects (`start`, `end`, `text`).

### Phase 4: Long-Video Chunking & Temporal Alignment
- Implement `temporal_alignment.py` to divide 30+ minute videos into ~5-minute chunks without losing timestamp information.

### Phase 5: Frame Sampling & Visual Feature Extraction
- Implement `frame_processor.py` and `visual_features.py` using CLIP / ResNet / MobileNet visual feature extractors and frame descriptions.

### Phase 6: TVSum & SumMe Dataset Preprocessing & Feature Extraction
- Implement `training/preprocess.py` and `training/extract_features.py` to parse TVSum & SumMe annotations (importance scores per shot/segment).
- Support downloading/synthesis of TVSum/SumMe benchmark format data.

### Phase 7: Multimodal Video Importance Model Training (PyTorch)
- Implement `training/model.py`: Multimodal fusion (Visual features + SentenceTransformer text embeddings -> Feature Fusion -> BiLSTM / Transformer Encoder -> Fully Connected Layer -> Importance Score [0, 1]).
- Implement `training/train.py` with video-level train/val/test splits (NO data leakage).
- Save trained checkpoint to `models/video_importance_model.pt`.

### Phase 8: Model Evaluation
- Implement `evaluation/evaluate_importance.py` and `evaluation/evaluate_summaries.py`.
- Calculate F-score, Precision, Recall against human annotations and output `evaluation/results.json`.

### Phase 9: Importance Model Integration into Long-Video Pipeline
- Implement `backend/services/importance_model.py` to load `video_importance_model.pt` and score 5-minute chunks and sub-segments to filter key content before LLM generation.

### Phase 10: Hierarchical LLM Summarization
- Implement `backend/services/summarizer.py` supporting API-based LLMs (e.g. OpenAI / Gemini / Groq / Anthropic or configurable local fallback).
- Chunk-level summary generation + Global LLM aggregation for final summary, key points, and timestamped sections.

### Phase 11: FastAPI Backend
- Implement `backend/main.py` & `backend/api/routes.py` with REST endpoints (`/api/upload`, `/api/process`, `/api/status/{task_id}`, `/api/result/{task_id}`).

### Phase 12: React Frontend UI
- Create React app in `frontend/` using Vite + TailwindCSS / CSS3 with video upload, progress bar, HTML5 video player with clickable timestamps, summary display, key points, and chapter sections.

### Phase 13: End-to-End System Integration
- Connect React frontend to FastAPI backend with CORS enabled and caching.

### Phase 14: 30+ Minute Video Execution & Test
- Run end-to-end pipeline test on a >30 minute test video.

### Phase 15: Runtime Error Verification & Optimization
- Test edge cases (no audio, corrupted video, long duration, CPU memory optimization).

### Phase 16: Documentation & Deliverables
- Generate `README.md`, `ARCHITECTURE.md`, `TRAINING.md`, `EVALUATION.md`, `API.md`, `.env.example`, and `TEST_REPORT.md`.
