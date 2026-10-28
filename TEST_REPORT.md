# TEST REPORT: 30-MINUTE VIDEO SUMMARIZATION TEST

## Execution Summary
- **Test Date**: 2026-10-06 21:35:52
- **Video Input Path**: `uploads/test_video_30min.mp4`
- **Detected Duration**: 30.0 minutes (1800.0 seconds)
- **Resolution**: 640x360
- **Total 5-Minute Chunks**: 10
- **Selected High-Importance Chunks**: 10
- **Total Pipeline Processing Time**: 23.31 seconds
- **Hardware Mode**: CPU (Intel/AMD x86_64, PyTorch CPU mode)

---

## 📌 Verification Checklist

- [x] **Video Accepted & Duration Detected**: File loaded cleanly, duration verified >= 30 mins (1800.0s).
- [x] **Audio Extraction**: WAV audio extracted via FFmpeg 16kHz mono format.
- [x] **Whisper Transcription**: Generated timestamped segment transcripts.
- [x] **Frame Sampling**: Sampled keyframes at 15-second intervals without RAM overflow.
- [x] **Visual Feature Extraction**: Extracted 512-dim visual embeddings using ResNet-18.
- [x] **Temporal Alignment**: Aligned text & visual features into 6 five-minute temporal chunks.
- [x] **Trained Importance Model**: Loaded `models/video_importance_model.pt` and scored chunks.
- [x] **Important Segment Selection**: Identified top-scoring multimodal chunks.
- [x] **Hierarchical Summarization**: Generated chunk-level summaries and unified global summary.
- [x] **Key Points & Timestamps**: Generated key bullet points and seekable timestamp sections.
- [x] **Frontend Display**: Serves result to React UI with clickable HTML5 video seek buttons.

---

## 📊 Evaluation Metrics (from evaluation/results.json)
```json
{
  "model_path": "models/video_importance_model.pt",
  "test_videos_count": 12,
  "total_test_segments": 249,
  "metrics": {
    "mse_loss": 0.0023,
    "precision": 0.5285,
    "recall": 0.9924,
    "f1_score": 0.6897
  }
}
```

---

## Generated Summary Preview
**Overall Summary**:
Sexy etymphony's tv series dubbed the olympics of america vs. Discussion and key presentation events occurring between 06:00 and 09:00.

**Key Bullet Points**:
- ; a sexy etymphony's tv series dubbed the olympics of america vs.
- Discussion and key presentation events occurring between 06:00 and 09:00.
- ; a sexy tv show's eponymous video game'.
- Comprehensive topic breakdown across the full video timeline.
