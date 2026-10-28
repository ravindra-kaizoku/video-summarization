# SYSTEM ARCHITECTURE

## Overview
The system implements a modular 6-stage pipeline designed specifically for long videos (30+ minutes minimum duration):

```
LONG VIDEO (>=30 MINS)
    │
    ├── Audio Branch ──> FFmpeg ──> Faster-Whisper ──> Timestamped Transcript
    │
    └── Visual Branch ──> OpenCV Frame Sampler (15s) ──> ResNet-18 (512-dim)
            │
            ▼
    Multimodal Temporal Alignment (5-Minute Chunks)
            │
            ▼
    Trained PyTorch Multimodal Video Importance Model
            │
            ▼
    Top-K Important Chunk Selection
            │
            ▼
    Hierarchical LLM Summarization (Chunk Summaries -> Global Summary)
            │
            ▼
    Final Output (Overall Summary + Key Points + Timestamped Sections)
```

## Detailed Component Specifications

### 1. Ingestion & Video Processing (`backend/services/video_processor.py`)
- Extracts total frames, FPS, duration, resolution, and audio track status using OpenCV and ImageIO-FFmpeg.

### 2. Audio Transcription (`backend/services/whisper_service.py`)
- Extracts 16kHz mono WAV file.
- Runs `faster-whisper` (`tiny` model) in int8 CPU precision.
- Preserves `start`, `end`, and `text` for every segment.

### 3. Visual Representation (`backend/services/visual_features.py`)
- Samples frames every `FRAME_INTERVAL_SECONDS=15`.
- Passes images through pre-trained ResNet-18 (fc layer replaced with Identity) to extract 512-dimensional feature vectors.

### 4. Multimodal Temporal Alignment (`backend/services/temporal_alignment.py`)
- Partitions the timeline into 5-minute chunks (`CHUNK_DURATION_SECONDS=300`).
- Combines transcript sentences and mean-pooled visual features into a unified sequence element.

### 5. Multimodal Importance Model (`training/model.py`)
- **Text Feature Dimension**: 384 (SentenceTransformer `all-MiniLM-L6-v2`)
- **Visual Feature Dimension**: 512 (ResNet-18)
- **Multimodal Fusion**: Project text & visual vectors into 128-dim embeddings, concatenate (256-dim), and pass through a 2-layer Bidirectional LSTM.
- **Output**: Sigmoid activation returning importance probability score $S \in [0, 1]$.

### 6. Hierarchical Summarization (`backend/services/summarizer.py`)
- Summarizes individual selected 5-minute chunks.
- Aggregates chunk summaries into a cohesive overall narrative, bulleted key points, and timestamped sections with seek seconds for the UI.
