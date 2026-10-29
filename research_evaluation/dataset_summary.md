# Dataset Audit and Verification Report
## Multimodal Long-Format Video Summarization Research

**Date:** 2026-10-29  
**Auditor:** Experimental Evaluation Specialist  
**Artifacts Generated:** `dataset_audit.csv`, `dataset_summary.md`

---

## 1. Executive Summary

This report documents the verification, structure, and integrity of all datasets present within the project. 

### Key Findings
1. **LVSum Dataset (Apple AI/ML Research):** Neither the raw videos, annotations, nor feature tensors from the LVSum dataset (`apple-aiml-research/ml-lvsum-video-summarization`) exist in the project repository or cache. The proposed plan of fine-tuning VASNet on LVSum was never executed.
2. **Current Training Dataset (`processed_dataset.json`):** Contains **75 video entries** (25 marked `SumMe`, 50 marked `TVSum`). However, inspection of `training/preprocess.py` confirms these features and ground-truth scores were generated **synthetically** using Gaussian noise vectors with temporal heuristics, rather than decoded from benchmark video frames.
3. **Real Video Assets in Production Cache:** The workspace contains **17 real long-format video files** in `uploads/` (totaling >1.8 GB, durations ranging from 20 seconds to 44.0 minutes). Real Faster-Whisper transcripts (up to 280 timestamped dialogue segments per video) and 512-dimensional ResNet-18 visual feature tensors are cached in `cache/transcripts/` and `cache/visual_features/`.
4. **Data Splitting Protocol:** Splitting is strictly **video-level** across 52 training videos (70%), 11 validation videos (15%), and 12 test videos (15%). No segment-level data leakage occurs across splits.

---

## 2. LVSum Dataset Investigation

The intended methodology proposed fine-tuning VASNet using the LVSum dataset:
- **Intended Source:** Apple Machine Learning Research — `https://github.com/apple-aiml-research/ml-lvsum-video-summarization`
- **Intended Purpose:** Long-video summarization benchmark (videos ranging from 10 to 60 minutes) with human-annotated summaries.

### Audit Findings:
- No files containing `"lvsum"` exist across any directory in the project.
- No repository clone, metadata CSV, or Apple AI/ML annotation format exists in the codebase.
- No network downloader or loader for LVSum was implemented in the `training/` directory.
- `training/download_dataset.py` instead points exclusively to a Kaggle dataset (`python16/summe-videos-and-tvsum-videos/versions/1`), which contains standard SumMe and TVSum files, not LVSum.

### Conclusion on LVSum:
The claim that the current system is trained or evaluated on LVSum is **empirically false**. Any IEEE research paper must state clearly that LVSum was not utilized in the current checkpoint, or must download the benchmark prior to final camera-ready publication.

---

## 3. Training Dataset Audit (`datasets/processed_dataset.json`)

The primary training file `datasets/processed_dataset.json` was audited programmatically. Summary statistics across all 75 entries:

| Attribute | SumMe Subset | TVSum Subset | Combined Total |
| :--- | :--- | :--- | :--- |
| **Total Video Count** | 25 | 50 | 75 |
| **Total Temporal Segments** | 496 | 1,010 | 1,506 |
| **Segments per Video (Mean ± SD)** | 19.8 ± 6.1 | 20.2 ± 5.9 | 20.1 ± 5.9 |
| **Segment Range** | 10 to 30 | 10 to 30 | 10 to 30 |
| **Visual Feature Dimension** | 512 | 512 | 512 |
| **Text Feature Dimension** | 384 | 384 | 384 |
| **Mean Annotation Score** | 0.8687 | 0.8781 | 0.8750 |
| **Score Standard Deviation** | 0.0463 | 0.0416 | 0.0435 |
| **Score Minimum** | 0.7608 | 0.7611 | 0.7608 |
| **Score Maximum** | 0.9500 | 0.9500 | 0.9500 |
| **Data Origin** | Synthetic Gaussian | Synthetic Gaussian | Synthetic Gaussian |

### Analysis of Synthetic Distribution
Inspection of `preprocess.py` lines 41–55 reveals the generator function:
```python
v = np.random.randn(n, 512).astype(np.float32)
v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
t = np.random.randn(n, 384).astype(np.float32)
t /= (np.linalg.norm(t, axis=1, keepdims=True) + 1e-8)
v_sig = np.std(v, axis=1) * 5.0
t_sig = np.sum(np.abs(t[:, :20]), axis=1)
raw = (0.3 + 0.35 * v_sig / (v_sig.max() + 1e-6) + 0.30 * t_sig / (t_sig.max() + 1e-6))
temporal_bias = np.abs(np.linspace(-0.1, 0.1, n))
scores = np.clip(raw + temporal_bias + np.random.normal(0, 0.03, n), 0.10, 0.95)
```
Because `v_sig` and `t_sig` have low variance for normalized Gaussian vectors, the generated scores cluster tightly between 0.76 and 0.95 (mean 0.875, standard deviation 0.043). This explains why the trained model predicts scores clustered narrowly around 0.86–0.88, resulting in near-zero correlation with human ranking.

---

## 4. Real Long-Video Assets (`uploads/` and `cache/`)

In contrast to the synthetic training set, the system's operational backend contains **real long-format videos** that were processed through the live application:

| Video Filename | Duration | Native Resolution | Frame Rate | Cached Transcript | Cached Visual Features |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `031452b8-8e0d-49d8-bd02-9dd99791826c.mp4` | 44.04 min | 854 × 480 | 25.0 fps | Yes (248 KB, 280+ segs) | Yes (2.48 MB, 177 frames) |
| `1f5308fe-49b9-4a05-a7ea-1e51c15e90c3.mkv` | 21.80 min | 1280 × 720 | 30.0 fps | Yes (57 KB) | Yes (1.23 MB) |
| `5f239bcd-c6a3-494a-b98c-22def1eb9744.mkv` | 43.50 min | 1280 × 720 | 30.0 fps | Yes (115 KB) | Yes (2.44 MB) |
| `e8b793cf-28a5-4eb1-a6b6-d2e4eb75e9c4.mp4` | 15.60 min | 1920 × 1080 | 29.97 fps | Yes (27 KB, 280 segs) | Yes (901 KB) |
| `f6c69748-7ce8-4cf9-a506-008112210c8e.mp4` | 14.80 min | 1920 × 1080 | 30.0 fps | Yes (26 KB) | Yes (855 KB) |
| `test_video_30min.mp4` | 28.00 min | 640 × 360 | 30.0 fps | Yes (11 KB) | Yes (1.67 MB) |

These files serve as an empirical testbed for evaluating inference latency, transcription fidelity, and chunk aggregation on actual long-duration content.

---

## 5. Splitting Protocol & Leakage Verification

The split recorded in `datasets/data_split.json` was cross-checked against `processed_dataset.json`:
- **Training Set:** 52 videos (35 TVSum, 17 SumMe) — 1,018 segments
- **Validation Set:** 11 videos (9 TVSum, 2 SumMe) — 239 segments
- **Test Set:** 12 videos (6 TVSum, 6 SumMe) — 249 segments
- **Intersection:** `Train ∩ Val = ∅`, `Train ∩ Test = ∅`, `Val ∩ Test = ∅`.
- **Segment Leakage:** 0 segments. The partition is strictly video-level.
