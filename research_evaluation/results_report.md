# Experimental Results and Empirical Evaluation Report
## A Multimodal Framework for Long-Format Video Summarization Using Visual Importance Estimation and Large Language Models

**Document Type:** IEEE-Style Results, Discussion, and Validation Section  
**Lead Evaluator:** Senior Machine Learning & Computer Vision Research Specialist  
**Evaluation Date:** October 29, 2026  
**Artifact Directory:** `research_evaluation/`  
**Dataset Audit:** [`dataset_audit.csv`](file:///c:/Users/ravindra_007/OneDrive/Desktop/ai%20prioritizer/video-summarizer/research_evaluation/dataset_audit.csv) | [`dataset_summary.md`](file:///c:/Users/ravindra_007/OneDrive/Desktop/ai%20prioritizer/video-summarizer/research_evaluation/dataset_summary.md)  
**Experimental Figures:** [`figures/`](file:///c:/Users/ravindra_007/OneDrive/Desktop/ai%20prioritizer/video-summarizer/research_evaluation/figures/)  
**Raw JSON Metrics:** [`results/`](file:///c:/Users/ravindra_007/OneDrive/Desktop/ai%20prioritizer/video-summarizer/research_evaluation/results/)

---

## 1. Executive Summary

This report delivers a rigorous, empirical evaluation of the existing video summarization system for publication in an IEEE-style peer-reviewed venue. Rather than assuming the architecture conformed to proposed research concepts, an exhaustive code and checkpoint audit was conducted, missing experimental harnesses were developed, and eight reproducible experiments were executed on CPU hardware.

### Key Empirical Findings:
1. **Architectural Divergence:** The operational pipeline does *not* employ GoogLeNet, VASNet, Kernel Temporal Segmentation (KTS), Qwen2.5-0.5B-Instruct, or SmolVLM-256M. Instead, it implements a **custom 2-layer Bidirectional LSTM (BiLSTM) fusion model (1.005M parameters)** with **ResNet-18 (512-dim visual features)**, **SentenceTransformers `all-MiniLM-L6-v2` (384-dim text features)**, **fixed 180-second chunking**, and **T5-small (60M parameters)** for transcript summarization.
2. **Multimodality Boundary:** The framework is multimodal *only* during segment importance scoring. Final abstractive text generation is strictly unimodal—T5-small receives speech transcript tokens and receives zero visual features or visual tokens.
3. **Training Data Provenance:** The 75 benchmark entries in `datasets/processed_dataset.json` were generated **synthetically** using normalized Gaussian random vectors with temporal bias heuristics. Neither raw SumMe/TVSum frames nor Apple's LVSum dataset were decoded during training.
4. **Predictive Performance on Test Data:** On the held-out test split (12 videos, 249 temporal segments), Checkpoint B achieves an **MSE of 0.0073** and an apparent **F1-score of 0.9939** at threshold 0.5. However, linear correlation is near zero (**Pearson $r = -0.0506, p = 0.4264$**, **Spearman $\rho = -0.0551$**). This reveals that the low MSE is an artifact of narrow target clustering (mean 0.875, std 0.043), rather than meaningful saliency ranking.
5. **Ablation Findings:** Ablation proves the multimodal architecture relies on feature co-presence. Zeroing out text degrades MSE by $4.6\times$ (0.0073 $\rightarrow$ 0.0335), zeroing out visual inputs degrades MSE by $5.8\times$ (0.0073 $\rightarrow$ 0.0423), and removing both causes catastrophic divergence (MSE 0.7588).
6. **Computational Efficiency:** On an AMD Ryzen 5 5600H CPU, segment scoring requires just **6.00 ms** per sequence. The end-to-end pipeline processes a 30-minute video in **11.75 seconds** (Real-Time Factor, $\text{RTF} = 0.0065$), demonstrating exceptional execution throughput for resource-constrained deployments.

---

## 2. Architecture and Implementation Audit

A systematic comparison between the proposed methodology and the actual codebase is summarized in Table I.

### TABLE I: Architectural Discrepancy Matrix

| Component | Proposed Methodology | Confirmed Code Implementation | Discrepancy Severity |
| :--- | :--- | :--- | :--- |
| **Visual Feature Extractor** | GoogLeNet / Inception-v1 (1024-dim) | Pretrained ResNet-18 (512-dim embedding) | **High** (Dimensionality & receptive field shift) |
| **Frame Sampling Rate** | $\approx 2.0$ frames per second (fps) | 1 frame per 15.0 seconds ($\approx 0.067$ fps in production) | **Critical** (30× temporal undersampling) |
| **Importance Model** | VASNet (self-attention mechanism) | Multimodal BiLSTM Fusion Network (2 layers, hidden=128) | **Critical** (Recurrent vs. transformer-attention) |
| **Temporal Segmentation** | Kernel Temporal Segmentation (KTS) | Fixed temporal windowing (180.0 seconds per chunk) | **High** (No content-adaptive scene boundaries) |
| **Segment Selection** | 0/1 Knapsack optimization | Coverage-aware windowed selection ($\le 15$ chunks) | **Moderate** (Heuristic vs. formal constraint) |
| **Speech Transcription** | Faster-Whisper | Faster-Whisper `tiny` (int8 quantized, CPU) | **Low** (Matches intended model family) |
| **Text Summarization** | Qwen2.5-0.5B-Instruct | HuggingFace T5-small (60M parameters) | **High** (Different model architecture & scale) |
| **Visual Captioning** | SmolVLM-256M-Instruct | Hardcoded string stub in `visual_features.py` | **Critical** (Feature omitted from execution) |
| **Final Multimodal Fusion** | Multimodal LLM integration | Transcript-only hierarchical aggregation via T5 | **Critical** (Visual data discarded prior to LLM) |

### 2.1 The Multimodal Boundary
A primary claim in multimodal research is that visual and acoustic modalities jointly condition summary generation. Inspection of `backend/services/summarizer.py` (lines 49–96 and 109–158) confirms:
- Chunks receive visual descriptions from `generate_visual_description()`, but this function returns a static string: `"Visual frame showing lecture slide or video scene activity."`
- In `generate_chunk_summary()`, only `chunk.get("transcript", "")` is fed into T5-small.
- In `generate_final_summary()`, the prompt combines previously generated textual chunk summaries.

**Formal Clarification for Publication:** The framework is *multimodally conditioned at the selection stage* (visual features inform which temporal chunks are retained), but *unimodal at the abstractive generation stage* (the generative text model processes transcripts exclusively).

---

## 3. Dataset Audit and Verification

### 3.1 Status of LVSum Benchmark
The research plan specified fine-tuning VASNet on the LVSum dataset. An audit across all directories, configuration files, and git history confirms:
- No LVSum video files, metadata annotations, or feature caches exist in the project repository.
- `training/download_dataset.py` references a Kaggle dataset (`python16/summe-videos-and-tvsum-videos/versions/1`), not LVSum.

### 3.2 Verification of Training Data (`datasets/processed_dataset.json`)
The primary training dataset consists of 75 videos totaling 1,506 temporal segments. Statistical inspection confirms this data was generated synthetically by `training/preprocess.py`:
- Features were generated using `np.random.randn(n, 512)` and `np.random.randn(n, 384)` with $L_2$ unit normalization.
- Ground-truth importance scores were calculated from feature variance plus temporal bias:
  $$\text{Score} = \text{clip}\left(0.30 + 0.35 \frac{\sigma_v}{\max(\sigma_v)} + 0.30 \frac{\sum |t_{:20}|}{\max(\sum |t_{:20}|)} + |\text{bias}| + \epsilon, 0.10, 0.95\right)$$
- Consequently, the ground-truth scores have an artificial mean of **0.8750** and standard deviation of **0.0435** (see [`dataset_audit.csv`](file:///c:/Users/ravindra_007/OneDrive/Desktop/ai%20prioritizer/video-summarizer/research_evaluation/dataset_audit.csv)).

### 3.3 Partitioning Protocol and Data Leakage
The data split in `datasets/data_split.json` establishes a strict video-level partition:
- **Training Set:** 52 videos (1,018 segments, 67.6%)
- **Validation Set:** 11 videos (239 segments, 15.9%)
- **Test Set:** 12 videos (249 segments, 16.5%)
- **Partition Verification:** $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$. Segment-level leakage is zero.

---

## 4. Quantitative Results & Model Performance

### 4.1 Evaluation Across Checkpoints
Table II reports the performance of the two saved PyTorch checkpoints evaluated on the 12 held-out test videos (249 segments).

### TABLE II: Test Set Evaluation Across Model Checkpoints

| Metric | Checkpoint A (`.pt`, Epoch 8) | Checkpoint B (`.pth`, Epoch 18) | Interpretation |
| :--- | :--- | :--- | :--- |
| **Mean Squared Error (MSE)** | **0.002266** | 0.007335 | Very low loss due to target clustering |
| **Root Mean Squared Error (RMSE)** | **0.047607** | 0.085646 | Average deviation $\approx 0.05$ to $0.08$ |
| **Mean Absolute Error (MAE)** | **0.038609** | 0.050162 | Low absolute error |
| **Pearson Correlation ($r$)** | -0.0581 ($p = 0.360$) | **-0.0506** ($p = 0.426$) | **No linear correlation with annotations** |
| **Spearman Rank Correlation ($\rho$)** | -0.0432 ($p = 0.498$) | **-0.0551** ($p = 0.386$) | **No monotonic ranking preservation** |
| **Kendall's Tau ($\tau$)** | -0.0295 ($p = 0.491$) | **-0.0378** ($p = 0.380$) | Ranking concordance is random |
| **Binary F1-Score ($\theta = 0.5$)** | **1.0000** | 0.9939 | Misleadingly high due to score clustering |
| **Precision ($\theta = 0.5$)** | 1.0000 | 1.0000 | All positive predictions are above threshold |
| **Recall ($\theta = 0.5$)** | 1.0000 | 0.9880 | Targets are almost entirely $> 0.5$ |
| **Prediction Standard Deviation** | 0.0168 | **0.0712** | Checkpoint B exhibits $4.2\times$ wider dynamic range |
| **Prediction Range $[\min, \max]$** | $[0.7964, 0.8906]$ | $[0.2349, 0.9056]$ | Checkpoint B breaks flat prediction mode |

As illustrated in Figure 1 ([`fig1_pred_vs_gt.svg`](file:///c:/Users/ravindra_007/OneDrive/Desktop/ai%20prioritizer/video-summarizer/research_evaluation/figures/fig1_pred_vs_gt.svg) / [`fig1_pred_vs_gt.png`](file:///c:/Users/ravindra_007/OneDrive/Desktop/ai%20prioritizer/video-summarizer/research_evaluation/figures/fig1_pred_vs_gt.png)), predictions cluster in a horizontal band around 0.86–0.88, failing to align with the $y = x$ reference diagonal.

### 4.2 Per-Dataset Type Breakdown
Evaluating Checkpoint B separately on the SumMe and TVSum test partitions reveals:
- **SumMe Test Set (6 videos, 117 segments):** $\text{MSE} = 0.007768$, Pearson $r = -0.1293$, $F_1 = 0.9957$.
- **TVSum Test Set (6 videos, 132 segments):** $\text{MSE} = 0.006952$, Pearson $r = +0.0126$, $F_1 = 0.9924$.

While TVSum achieves a slightly lower MSE and non-negative correlation, neither partition demonstrates statistically meaningful saliency prediction ($p > 0.1$ for both).

---

## 5. Ablation Studies

To evaluate the contribution of individual modalities, we conducted a systematic ablation study by zeroing feature representations at inference time without retraining. Results are presented in Table III and Figure 3 ([`fig3_ablation.svg`](file:///c:/Users/ravindra_007/OneDrive/Desktop/ai%20prioritizer/video-summarizer/research_evaluation/figures/fig3_ablation.svg)).

### TABLE III: Modality Ablation Results on Test Set

| Configuration | Visual Input (512-dim) | Text Input (384-dim) | MSE Loss | Pearson $r$ | Pred Mean | Pred Std | Binary F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Multimodal (Full)** | Active | Active | **0.007335** | -0.0506 | 0.8668 | 0.0712 | **0.9939** |
| **Visual-Only** | Active | Zeroed | 0.042302 | -0.1034 | 0.7302 | 0.1684 | 0.9168 |
| **Text-Only** | Zeroed | Active | 0.033519 | **-0.0064** | 0.7495 | 0.1533 | 0.9399 |
| **Zero Baseline (Ablated)**| Zeroed | Zeroed | 0.758825 | +0.0182 | 0.0041 | 0.0061 | 0.0000 |

### Key Ablation Insights:
1. **Multimodal Synergy:** Fusing both modalities yields the lowest MSE ($0.0073$). Removing either visual or text features increases error by **$5.7\times$** and **$4.6\times$**, respectively.
2. **Text Dominance Over Visual:** Text-only features achieve a lower error ($0.0335$) than visual-only features ($0.0423$), indicating the fusion MLP places greater reliance on the SentenceTransformer projection.
3. **Catastrophic Failure Without Input:** When both inputs are ablated, output scores collapse to near-zero ($0.0041$), resulting in an MSE of $0.7588$ and $F_1 = 0.0000$. This confirms the model does not merely output a hardcoded bias.

---

## 6. Cross-Split Generalization Analysis

To test for overfitting, the trained model was evaluated across all three partitions (Table IV and Figure 4).

### TABLE IV: Generalization Across Data Splits

| Partition | Video Count | Segment Count | MSE Loss | Pearson $r$ | Pred Mean | Pred Std | Target Std |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Training Split** | 52 | 1,018 | **0.002103** | **+0.3010** | 0.8804 | 0.0199 | 0.0477 |
| **Validation Split** | 11 | 239 | 0.003752 | -0.0570 | 0.8732 | 0.0381 | 0.0458 |
| **Test Split** | 12 | 249 | 0.007335 | -0.0506 | 0.8668 | 0.0712 | 0.0433 |

### Findings:
- On the training set, the model achieves a statistically significant positive correlation (**$r = +0.3010, p < 10^{-23}$**) with low loss ($0.0021$).
- However, on both validation and test sets, the correlation collapses to negative values ($-0.057$ and $-0.051$), and MSE increases by **$3.5\times$**.
- This indicates **overfitting to synthetic training artifacts**: the BiLSTM memorized temporal position patterns specific to the training set that failed to generalize to unseen sequences.

---

## 7. Computational Complexity and Latency Profiling

### 7.1 Model Architectural Footprint
As detailed in [`fig7_param_distribution.svg`](file:///c:/Users/ravindra_007/OneDrive/Desktop/ai%20prioritizer/video-summarizer/research_evaluation/figures/fig7_param_distribution.svg) and Table V, the model contains **1,004,801 parameters**, occupying **3.83 MB** in 32-bit floating point format.

### TABLE V: Parameter Distribution Across Architecture

| Sub-Module | Layer Specification | Parameter Count | Percentage |
| :--- | :--- | :--- | :--- |
| **Visual Projection** | $\text{Linear}(512 \rightarrow 128) + \text{BN1d}(128)$ | 66,048 | 6.6% |
| **Text Projection** | $\text{Linear}(384 \rightarrow 128) + \text{BN1d}(128)$ | 49,664 | 4.9% |
| **Multimodal Fusion**| $\text{Linear}(256 \rightarrow 256)$ | 65,792 | 6.5% |
| **BiLSTM Sequence Model**| 2 Layers, $\text{input}=256, \text{hidden}=128$ (Bidirectional) | 789,504 | 78.6% |
| **Saliency Classifier**| $\text{Linear}(256 \rightarrow 128) + \text{Linear}(128 \rightarrow 1)$ | 33,025 | 3.3% |
| **Total** | Full Multimodal Network | **1,004,801** | **100.0%** |

### 7.2 End-to-End Execution Latency on Real Long Videos
The complete pipeline was benchmarked on real long-format video assets hosted in `uploads/` using cached transcripts and visual features (Table VI).

### TABLE VI: Pipeline Stage Latency Breakdown (AMD Ryzen 5 5600H CPU)

| Video Name | Duration | Total Chunks | Scoring & Selection Latency | T5-small Summarization | Total Pipeline Time | Real-Time Factor (RTF) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `diagnostic_test_001.mp4` | 0.3 min | 1 | 16.0 ms | 1.30 s | 1.32 s | 0.0698 |
| `e8b793cf-...-e9c4.mp4` | 15.8 min | 6 | 196.4 ms | 8.40 s | 8.64 s | 0.0091 |
| `test_video_30min.mp4` | 30.0 min | 10 | 338.6 ms | 11.37 s | 11.75 s | **0.0065** |

### Latency Highlights:
1. **Sub-Second Importance Scoring:** Scoring an entire 30-minute video (10 chunks) takes only **338.6 ms** on CPU, averaging **33.8 ms per 3-minute chunk**.
2. **Faster-Than-Real-Time Operation:** The entire summarization pipeline achieves an RTF of **0.0065** for a 30-minute video. This indicates the system runs over **150× faster than real-time playback**.
3. **Bottleneck Identification:** T5-small abstractive text generation accounts for **96.8%** of total runtime ($11.37\text{s} / 11.75\text{s}$), whereas importance scoring and temporal alignment account for less than 3.2%.

---

## 8. Hierarchical Text Summarization Quality

The text summarization module was evaluated on real and benchmark transcripts (Table VII).

### TABLE VII: T5-Small Abstractive Summarization Benchmarks

| Sample ID | Input Word Count | Output Summary Words | Compression Ratio | Latency (s) | Representative Abstractive Summary Output |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Chunk 1 (CV Lecture)** | 34 | 9 | **73.5%** | 1.020 s | *"The architecture allows hierarchical feature extraction from low-level edges."* |
| **Chunk 2 (DB Systems)** | 30 | 19 | **36.7%** | 0.867 s | *"We cover database indexing and query optimization. B-trees provide logarithmic search guarantees..."* |
| **Chunk 3 (Evaluation)** | 30 | 23 | **23.3%** | 0.530 s | *"The experimental results demonstrate that our proposed multimodal fusion model reduces MSE significantly..."* |
| **Overall Video Aggregation** | 245 | 48 | **80.4%** | 2.150 s | Executive overview with 6 structured bullet points and timestamped section headers. |

T5-small produces clean, grammatically coherent sentences with an average compression ratio of **44.5%** for individual chunks and over **80%** for final video summaries.

---

## 9. Comparison with Published Literature Baselines

To contextualize the findings within the broader video summarization literature, Table VIII benchmarks the framework against state-of-the-art models on SumMe and TVSum.

### TABLE VIII: Comparison with Published Literature on SumMe and TVSum

| Method | Backbone Architecture | Modality | SumMe F1 (%) | TVSum F1 (%) |
| :--- | :--- | :---: | :---: | :---: |
| **dppLSTM** (Zhang et al., ECCV 2016) | GoogLeNet (1024-d) + BiLSTM + DPP | Visual | 38.6 | 54.7 |
| **DR-DSN** (Zhou et al., AAAI 2018) | GoogLeNet (1024-d) + BiLSTM + RL | Visual | 41.4 | 57.6 |
| **VASNet** (Fajtl et al., AMDO 2019) | GoogLeNet (1024-d) + Self-Attention | Visual | 49.7 | 61.4 |
| **PGL-SUM** (Apostolidis et al., ACM MM 2021)| Inception-v3 + Multi-Head Attention | Visual | 50.8 | 63.9 |
| **SUM-GNN** (Zhao et al., TMM 2022) | ResNet-50 + Graph Neural Network | Visual | 51.9 | 64.3 |
| **This Work (Reported Binary Metric)** | ResNet-18 (512-d) + ST (384-d) + BiLSTM | Multimodal | **99.6\*** | **99.2\*** |
| **This Work (Rank Correlation $r$)** | ResNet-18 (512-d) + ST (384-d) + BiLSTM | Multimodal | **-0.129** | **+0.013** |

> **\*Scientific Caveat for Publication:** The $99\%+$ binary F1-score reported by this framework is an artifact of thresholding clustered synthetic data at $\theta = 0.5$. In contrast, published literature baselines evaluate the intersection-over-union (IoU) of selected video segments against multiple human user summaries constrained to a 15% video length budget. Under formal rank correlation, the current checkpoint does not yet surpass random ranking on benchmark annotations.

---

## 10. Threats to Validity and Failure Modes

1. **Synthetic Training Artifacts:** The model was trained on synthetic Gaussian vectors rather than real benchmark frames. This explains the disconnect between training correlation ($r = +0.30$) and test correlation ($r = -0.05$).
2. **Score-Scaling Heuristic:** In `importance_model.py` (lines 120–131), when prediction variance drops below 0.15, the application artificially recalculates scores between 0.42 and 0.96 using transcript word count and frame standard deviation:
   $$\text{Score}_{\text{scaled}} = \min\left(0.96, \max\left(0.42, 0.50 + 0.35 \frac{i}{N-1} + \Delta_{\text{score}} + \frac{W_t}{200} + 2\sigma_v\right)\right)$$
   This heuristic masks the model's low variance in production, but represents an engineering workaround rather than a learned capability.
3. **Coarse Temporal Granularity:** Fixed 180-second chunks lack boundary awareness, occasionally grouping distinct topics into a single segment.

---

## 11. Concrete Recommendations for Future Research

1. **Execute Real Feature Extraction:** Run `python -m training.feature_extraction` on real SumMe and TVSum videos (or LVSum) using Kaggle credentials to replace synthetic data with authentic ImageNet and human annotations.
2. **Implement KTS or Semantic Scene Boundaries:** Replace the rigid 180-second chunking with Kernel Temporal Segmentation or optical flow scene-change detection.
3. **Transition to Cross-Attention Summarization:** Pass visual features directly to the LLM (e.g., using SmolVLM or Qwen2-VL) so that visual demonstrations directly shape the generated text summary.
4. **Adopt Standard Evaluation Protocols:** Evaluate using the knapsack 15% video budget and 5-fold cross-validation with harmonic mean F1 against human summary selections.
