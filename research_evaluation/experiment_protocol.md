# Experimental Evaluation Protocol
## A Multimodal Framework for Long-Format Video Summarization Using Visual Importance Estimation and Large Language Models

**Document Version:** 1.0  
**Target Venue:** IEEE Transactions / IEEE Conference on Multimedia / CVPR Workshops  
**Evaluation Lead:** Experimental Evaluation Specialist  
**Location:** `research_evaluation/experiment_protocol.md`

---

## 1. Research Objectives and Evaluation Scope

This protocol establishes the formal experimental methodology to evaluate the video summarization framework.

### Primary Objectives:
1. **Model Fidelity and Accuracy:** Quantify the accuracy of the multimodal video importance model in predicting segment-level saliency.
2. **Correlation Analysis:** Measure linear and monotonic rank correlation between predicted importance and human ground-truth rankings.
3. **Modality Ablation:** Assess the marginal contribution of visual features (ResNet-18) versus textual features (SentenceTransformers) in importance scoring.
4. **Generalization Analysis:** Evaluate performance divergence across independent train, validation, and test splits to detect overfitting or dataset shift.
5. **Computational Complexity & Latency:** Profile latency, memory footprint, and processing bottlenecks across all pipeline stages (video decoding, feature extraction, STT transcription, chunk alignment, scoring, LLM summarization).
6. **Abstractive Summary Quality:** Evaluate the text summarization module (T5-small hierarchical summarizer) on coherence, length compression, and key takeaways.

---

## 2. Experimental Metrics and Mathematical Definitions

### 2.1 Regression Metrics
Let $y_i$ denote the ground-truth human importance score for segment $i$, and $\hat{y}_i$ denote the model's predicted score, for $i = 1, \dots, N$.

1. **Mean Squared Error (MSE):**
   $$\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2$$

2. **Root Mean Squared Error (RMSE):**
   $$\text{RMSE} = \sqrt{\text{MSE}}$$

3. **Mean Absolute Error (MAE):**
   $$\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |\hat{y}_i - y_i|$$

### 2.2 Correlation and Ranking Metrics
1. **Pearson Correlation Coefficient ($r$):**
   $$r = \frac{\sum_{i=1}^{N} (\hat{y}_i - \bar{\hat{y}})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{N} (\hat{y}_i - \bar{\hat{y}})^2} \sqrt{\sum_{i=1}^{N} (y_i - \bar{y})^2}}$$
   Measures linear correlation between predictions and ground-truth. Range: $[-1, +1]$.

2. **Spearman Rank Correlation ($\rho$):**
   $$\rho = 1 - \frac{6 \sum_{i=1}^N d_i^2}{N(N^2 - 1)}$$
   Where $d_i = \text{rank}(\hat{y}_i) - \text{rank}(y_i)$. Measures monotonic ranking preservation.

3. **Kendall's Tau ($\tau$):**
   $$\tau = \frac{C - D}{\frac{1}{2} N(N - 1)}$$
   Where $C$ is the count of concordant pairs and $D$ is the count of discordant pairs.

### 2.3 Binary Classification Metrics (Thresholded Selection)
For a selection threshold $\theta \in [0, 1]$ (default $\theta = 0.5$):
$$\hat{b}_i = \mathbb{I}(\hat{y}_i \ge \theta), \quad b_i = \mathbb{I}(y_i \ge \theta)$$

1. **Precision:** $P = \frac{\text{TP}}{\text{TP} + \text{FP}}$
2. **Recall:** $R = \frac{\text{TP}}{\text{TP} + \text{FN}}$
3. **F1-Score:** $F_1 = \frac{2 \cdot P \cdot R}{P + R}$
4. **Accuracy:** $\text{Acc} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$

### 2.4 Computational Metrics
1. **Wall-clock Latency (ms or seconds):** Total execution time per pipeline component.
2. **Real-Time Factor (RTF):**
   $$\text{RTF} = \frac{\text{Processing Time}}{\text{Video Duration}}$$
   $\text{RTF} < 1.0$ indicates faster-than-real-time throughput.
3. **Parameter Count and Disk Footprint:** Model parameters in millions (M) and checkpoint size in megabytes (MB).

---

## 3. Experimental Design and Testbed

### 3.1 Hardware & Software Environment
- **Host CPU:** AMD Ryzen 5 5600H (6 Cores / 12 Threads @ 3.3 GHz base, 4.2 GHz boost)
- **RAM:** 16 GB DDR4
- **Operating System:** Windows 11 Home 64-bit
- **Python Environment:** Python 3.10.11, PyTorch 2.11.0 (CPU), HuggingFace Transformers 5.17.0, Faster-Whisper 1.2.1

### 3.2 Models Evaluated
1. **Model Checkpoint A (`models/video_importance_model.pt`):**
   - Training: Epoch 8 checkpoint
   - Architecture: MultimodalImportanceModel (BiLSTM 2 layers, 1.005M parameters)
   - Val Loss: 0.001107
2. **Model Checkpoint B (`models/video_importance_model.pth`):**
   - Training: Epoch 18 checkpoint (early stopping at epoch 26)
   - Val Loss: 0.000447

### 3.3 Experiments Planned & Executed
| Experiment ID | Title | Purpose |
| :--- | :--- | :--- |
| **EXP-1** | Checkpoint Evaluation | Compare Checkpoints A and B on held-out test split |
| **EXP-2** | Per-Dataset Type Breakdown | Analyze SumMe vs. TVSum performance independently |
| **EXP-3** | Modality Ablation | Zero-out visual or text inputs to isolate modality contributions |
| **EXP-4** | Cross-Split Generalization | Assess generalization gap across Train, Val, and Test splits |
| **EXP-5** | Score Distribution Analysis | Examine empirical distributions, percentiles, and residuals |
| **EXP-6** | Architectural Complexity | Profile parameter allocation across layers |
| **EXP-7** | Inference Timing | Measure per-segment forward latency (10 repeated trials) |
| **EXP-8** | Threshold Sensitivity | Sweep classification threshold $\theta \in [0.1, 0.95]$ |
| **EXP-9** | End-to-End Pipeline Profiling | Benchmark wall-clock latency per stage on long-format video |
| **EXP-10** | Text Summarization Analysis | Evaluate abstractive compression, length, and coherence |
