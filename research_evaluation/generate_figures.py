"""
research_evaluation/generate_figures.py
=======================================
Generates publication-ready figures and tables for the IEEE-style results section.

Usage (from project root):
    python -m research_evaluation.generate_figures
"""

import os
import sys
import json
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "research_evaluation", "results")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "research_evaluation", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# IEEE-style formatting
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})


def fig1_pred_vs_gt_scatter():
    """Figure 1: Predicted vs Ground-Truth importance scores scatter plot."""
    with open(os.path.join(RESULTS_DIR, "exp5_score_distributions.json")) as f:
        data = json.load(f)

    preds = np.array(data["raw_predictions"])
    tgts = np.array(data["raw_targets"])

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.scatter(tgts, preds, alpha=0.5, s=20, c='#2196F3', edgecolors='none')
    ax.plot([0, 1], [0, 1], 'r--', linewidth=1.0, label='Ideal (y=x)')
    ax.set_xlabel('Ground-Truth Importance Score')
    ax.set_ylabel('Predicted Importance Score')
    ax.set_title('Predicted vs. Ground-Truth Scores')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # Add Pearson r annotation
    from scipy.stats import pearsonr
    r, p = pearsonr(preds, tgts)
    ax.text(0.05, 0.92, f'Pearson r = {r:.4f}\np = {p:.4f}',
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    path = os.path.join(FIGURES_DIR, "fig1_pred_vs_gt.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def fig2_score_distributions():
    """Figure 2: Histogram comparison of predicted vs ground-truth score distributions."""
    with open(os.path.join(RESULTS_DIR, "exp5_score_distributions.json")) as f:
        data = json.load(f)

    preds = np.array(data["raw_predictions"])
    tgts = np.array(data["raw_targets"])

    fig, ax = plt.subplots(figsize=(5, 3.5))
    bins = np.arange(0, 1.05, 0.05)
    ax.hist(tgts, bins=bins, alpha=0.6, label='Ground-Truth', color='#4CAF50', edgecolor='black', linewidth=0.5)
    ax.hist(preds, bins=bins, alpha=0.6, label='Predictions', color='#FF9800', edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Importance Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Importance Scores')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    path = os.path.join(FIGURES_DIR, "fig2_score_distributions.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def fig3_ablation_bar_chart():
    """Figure 3: Ablation study bar chart comparing MSE and Pearson r across modalities."""
    with open(os.path.join(RESULTS_DIR, "exp3_ablation.json")) as f:
        data = json.load(f)

    conditions = ['Multimodal', 'Visual Only', 'Text Only', 'No Input']
    keys = ['multimodal', 'visual_only', 'text_only', 'no_input']
    mse_vals = [data[k]["metrics"]["mse"] for k in keys]
    pearson_vals = [data[k]["metrics"]["pearson_r"] for k in keys]
    pred_std_vals = [data[k]["metrics"]["pred_std"] for k in keys]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    # MSE
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#F44336']
    bars = axes[0].bar(conditions, mse_vals, color=colors, edgecolor='black', linewidth=0.5)
    axes[0].set_ylabel('MSE')
    axes[0].set_title('(a) Mean Squared Error')
    axes[0].tick_params(axis='x', rotation=25)
    for bar, val in zip(bars, mse_vals):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{val:.4f}', ha='center', va='bottom', fontsize=8)

    # Pearson r
    bars = axes[1].bar(conditions, pearson_vals, color=colors, edgecolor='black', linewidth=0.5)
    axes[1].set_ylabel('Pearson r')
    axes[1].set_title('(b) Correlation')
    axes[1].tick_params(axis='x', rotation=25)
    axes[1].axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    for bar, val in zip(bars, pearson_vals):
        offset = 0.005 if val >= 0 else -0.02
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
                     f'{val:.4f}', ha='center', va='bottom', fontsize=8)

    # Prediction Std
    bars = axes[2].bar(conditions, pred_std_vals, color=colors, edgecolor='black', linewidth=0.5)
    axes[2].set_ylabel('Prediction Std Dev')
    axes[2].set_title('(c) Score Diversity')
    axes[2].tick_params(axis='x', rotation=25)
    for bar, val in zip(bars, pred_std_vals):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                     f'{val:.4f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig3_ablation.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def fig4_cross_split_comparison():
    """Figure 4: Cross-split generalization comparison."""
    with open(os.path.join(RESULTS_DIR, "exp4_cross_split.json")) as f:
        data = json.load(f)

    splits = ['Train', 'Validation', 'Test']
    keys = ['train', 'val', 'test']
    mse_vals = [data[k]["metrics"]["mse"] for k in keys]
    pearson_vals = [data[k]["metrics"]["pearson_r"] for k in keys]
    pred_std = [data[k]["metrics"]["pred_std"] for k in keys]
    tgt_std = [data[k]["metrics"]["target_std"] for k in keys]

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))

    x = np.arange(len(splits))
    width = 0.35

    # MSE comparison
    colors = ['#2196F3', '#FF9800', '#F44336']
    bars = axes[0].bar(splits, mse_vals, color=colors, edgecolor='black', linewidth=0.5)
    axes[0].set_ylabel('MSE')
    axes[0].set_title('(a) MSE by Data Split')
    for bar, val in zip(bars, mse_vals):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0002,
                     f'{val:.4f}', ha='center', va='bottom', fontsize=8)

    # Std comparison
    bars1 = axes[1].bar(x - width/2, pred_std, width, label='Predictions', color='#FF9800', edgecolor='black', linewidth=0.5)
    bars2 = axes[1].bar(x + width/2, tgt_std, width, label='Ground Truth', color='#4CAF50', edgecolor='black', linewidth=0.5)
    axes[1].set_ylabel('Standard Deviation')
    axes[1].set_title('(b) Score Variability')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(splits)
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig4_cross_split.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def fig5_threshold_sensitivity():
    """Figure 5: F1, Precision, Recall vs threshold."""
    with open(os.path.join(RESULTS_DIR, "exp8_threshold_sensitivity.json")) as f:
        data = json.load(f)

    thresholds = [t["threshold"] for t in data["threshold_analysis"]]
    f1_vals = [t["f1_score"] for t in data["threshold_analysis"]]
    prec_vals = [t["precision"] for t in data["threshold_analysis"]]
    rec_vals = [t["recall"] for t in data["threshold_analysis"]]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(thresholds, f1_vals, 'b-o', markersize=4, label='F1 Score', linewidth=1.5)
    ax.plot(thresholds, prec_vals, 'g--s', markersize=3, label='Precision', linewidth=1.0)
    ax.plot(thresholds, rec_vals, 'r--^', markersize=3, label='Recall', linewidth=1.0)
    ax.set_xlabel('Classification Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Binary Classification Metrics vs. Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.05, 1.0)
    ax.set_ylim(-0.05, 1.05)

    path = os.path.join(FIGURES_DIR, "fig5_threshold_sensitivity.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def fig6_per_video_analysis():
    """Figure 6: Per-video MSE and Pearson r heatmap-like bar chart."""
    with open(os.path.join(RESULTS_DIR, "exp1_model_evaluation.json")) as f:
        data = json.load(f)

    pth_data = data["pth"]["per_video_results"]
    video_ids = [v["video_id"].replace("_video_", " ") for v in pth_data]
    mse_vals = [v["metrics"]["mse"] for v in pth_data]
    pearson_vals = [v["metrics"]["pearson_r"] for v in pth_data]

    fig, axes = plt.subplots(2, 1, figsize=(7, 5))

    colors_mse = ['#F44336' if m > 0.01 else '#FF9800' if m > 0.005 else '#4CAF50' for m in mse_vals]
    bars = axes[0].barh(video_ids, mse_vals, color=colors_mse, edgecolor='black', linewidth=0.5)
    axes[0].set_xlabel('MSE')
    axes[0].set_title('(a) Per-Video MSE')
    axes[0].invert_yaxis()

    colors_r = ['#F44336' if r < -0.1 else '#FF9800' if r < 0.1 else '#4CAF50' for r in pearson_vals]
    bars = axes[1].barh(video_ids, pearson_vals, color=colors_r, edgecolor='black', linewidth=0.5)
    axes[1].set_xlabel('Pearson r')
    axes[1].set_title('(b) Per-Video Correlation')
    axes[1].axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
    axes[1].invert_yaxis()

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig6_per_video.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def fig7_architecture_diagram():
    """Figure 7: Model architecture parameter distribution pie chart."""
    with open(os.path.join(RESULTS_DIR, "exp6_model_architecture.json")) as f:
        data = json.load(f)

    # Group parameters by component
    components = {
        "Visual Projection": 0,
        "Text Projection": 0,
        "Fusion MLP": 0,
        "BiLSTM": 0,
        "Classifier": 0,
    }

    for layer in data["layer_details"]:
        name = layer["name"]
        params = layer["params"]
        if "visual_proj" in name:
            components["Visual Projection"] += params
        elif "text_proj" in name:
            components["Text Projection"] += params
        elif "fusion_fc" in name:
            components["Fusion MLP"] += params
        elif "bilstm" in name:
            components["BiLSTM"] += params
        elif "classifier" in name:
            components["Classifier"] += params

    labels = list(components.keys())
    sizes = list(components.values())
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']

    fig, ax = plt.subplots(figsize=(5, 4))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct='%1.1f%%',
        startangle=90, pctdistance=0.85,
        wedgeprops=dict(linewidth=1, edgecolor='white')
    )
    for t in autotexts:
        t.set_fontsize(8)
    ax.set_title(f'Parameter Distribution ({data["total_parameters"]:,} total)')

    path = os.path.join(FIGURES_DIR, "fig7_param_distribution.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def fig8_residual_analysis():
    """Figure 8: Residual (prediction error) analysis."""
    with open(os.path.join(RESULTS_DIR, "exp5_score_distributions.json")) as f:
        data = json.load(f)

    preds = np.array(data["raw_predictions"])
    tgts = np.array(data["raw_targets"])
    residuals = preds - tgts

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    # Residual histogram
    axes[0].hist(residuals, bins=30, color='#2196F3', edgecolor='black', linewidth=0.5, alpha=0.7)
    axes[0].axvline(x=0, color='red', linestyle='--', linewidth=1.0)
    axes[0].set_xlabel('Residual (Predicted − Ground Truth)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('(a) Residual Distribution')
    axes[0].text(0.05, 0.92, f'μ = {residuals.mean():.4f}\nσ = {residuals.std():.4f}',
                 transform=axes[0].transAxes, fontsize=9,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Residual vs ground truth
    axes[1].scatter(tgts, residuals, alpha=0.5, s=20, c='#FF9800', edgecolors='none')
    axes[1].axhline(y=0, color='red', linestyle='--', linewidth=1.0)
    axes[1].set_xlabel('Ground-Truth Score')
    axes[1].set_ylabel('Residual')
    axes[1].set_title('(b) Residuals vs. Ground Truth')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig8_residuals.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def generate_latex_tables():
    """Generate LaTeX-formatted tables for the IEEE paper."""

    # Table 1: Model Comparison (both checkpoints)
    with open(os.path.join(RESULTS_DIR, "exp1_model_evaluation.json")) as f:
        exp1 = json.load(f)

    table1 = r"""
\begin{table}[t]
\caption{Importance Model Evaluation on Test Set}
\label{tab:model_eval}
\centering
\begin{tabular}{lcc}
\hline
\textbf{Metric} & \textbf{Checkpoint A (E8)} & \textbf{Checkpoint B (E18)} \\
\hline
"""
    metrics_list = ["mse", "rmse", "mae", "pearson_r", "spearman_r", "kendall_tau", "f1_score", "precision", "recall"]
    metric_names = ["MSE", "RMSE", "MAE", "Pearson $r$", "Spearman $\\rho$", "Kendall $\\tau$", "F1 Score", "Precision", "Recall"]

    for mname, mkey in zip(metric_names, metrics_list):
        v1 = exp1["pt"]["aggregate_metrics"][mkey]
        v2 = exp1["pth"]["aggregate_metrics"][mkey]
        table1 += f"{mname} & {v1:.4f} & {v2:.4f} \\\\\n"

    table1 += r"""\hline
\end{tabular}
\end{table}
"""

    # Table 2: Ablation Study
    with open(os.path.join(RESULTS_DIR, "exp3_ablation.json")) as f:
        exp3 = json.load(f)

    table2 = r"""
\begin{table}[t]
\caption{Modality Ablation Study}
\label{tab:ablation}
\centering
\begin{tabular}{lcccc}
\hline
\textbf{Condition} & \textbf{MSE} & \textbf{Pearson $r$} & \textbf{Pred Std} & \textbf{F1} \\
\hline
"""
    cond_names = {"multimodal": "Multimodal (V+T)", "visual_only": "Visual Only", "text_only": "Text Only", "no_input": "No Input (Baseline)"}
    for key, name in cond_names.items():
        m = exp3[key]["metrics"]
        table2 += f"{name} & {m['mse']:.4f} & {m['pearson_r']:.4f} & {m['pred_std']:.4f} & {m['f1_score']:.4f} \\\\\n"

    table2 += r"""\hline
\end{tabular}
\end{table}
"""

    # Table 3: Cross-Split Generalization
    with open(os.path.join(RESULTS_DIR, "exp4_cross_split.json")) as f:
        exp4 = json.load(f)

    table3 = r"""
\begin{table}[t]
\caption{Cross-Split Generalization Analysis}
\label{tab:crosssplit}
\centering
\begin{tabular}{lccccc}
\hline
\textbf{Split} & \textbf{Videos} & \textbf{Segments} & \textbf{MSE} & \textbf{Pearson $r$} & \textbf{Pred Std} \\
\hline
"""
    split_names = {"train": "Training", "val": "Validation", "test": "Test"}
    for key, name in split_names.items():
        d = exp4[key]
        m = d["metrics"]
        table3 += f"{name} & {d['num_videos']} & {d['num_segments']} & {m['mse']:.4f} & {m['pearson_r']:.4f} & {m['pred_std']:.4f} \\\\\n"

    table3 += r"""\hline
\end{tabular}
\end{table}
"""

    # Table 4: Architecture Summary
    with open(os.path.join(RESULTS_DIR, "exp6_model_architecture.json")) as f:
        exp6 = json.load(f)

    table4 = r"""
\begin{table}[t]
\caption{Model Architecture Summary}
\label{tab:architecture}
\centering
\begin{tabular}{ll}
\hline
\textbf{Component} & \textbf{Configuration} \\
\hline
Visual Projection & Linear(512$\rightarrow$128) + BN + ReLU + Dropout(0.2) \\
Text Projection & Linear(384$\rightarrow$128) + BN + ReLU + Dropout(0.2) \\
Fusion & Concatenation(256) + MLP(256$\rightarrow$256) \\
Temporal Encoder & BiLSTM (hidden=128, layers=2) \\
Classifier & MLP(256$\rightarrow$128$\rightarrow$1) + Sigmoid \\
\hline
Total Parameters & """ + f"{exp6['total_parameters']:,}" + r""" \\
Model Size & """ + f"{exp6['model_size_mb']:.2f} MB" + r""" \\
\hline
\end{tabular}
\end{table}
"""

    tables = table1 + "\n" + table2 + "\n" + table3 + "\n" + table4
    path = os.path.join(FIGURES_DIR, "latex_tables.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write(tables)
    print(f"  Saved: {path}")


def main():
    print("=" * 60)
    print("GENERATING PUBLICATION-READY FIGURES AND TABLES")
    print("=" * 60)

    fig1_pred_vs_gt_scatter()
    fig2_score_distributions()
    fig3_ablation_bar_chart()
    fig4_cross_split_comparison()
    fig5_threshold_sensitivity()
    fig6_per_video_analysis()
    fig7_architecture_diagram()
    fig8_residual_analysis()
    generate_latex_tables()

    print(f"\nAll figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
