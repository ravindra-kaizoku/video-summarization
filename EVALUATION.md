# MODEL EVALUATION REPORT

## Evaluated Checkpoint
`models/video_importance_model.pt`

## Results (`evaluation/results.json`)
Evaluated on 12 held-out test videos (never seen during training).

| Metric | Score | Explanation |
| :--- | :--- | :--- |
| **MSE Loss** | **0.0023** | Low mean squared error against ground-truth human annotations |
| **Precision** | **0.5285** | Proportion of predicted key segments that match human annotations |
| **Recall** | **0.9924** | Proportion of actual human key segments captured by the model |
| **F1-Score** | **0.6897** | Harmonic mean of Precision and Recall |

---

## Benchmark Source
Human importance annotations from **SumMe** and **TVSum** benchmark formats.
Evaluation script location: `evaluation/evaluate_importance.py`.
