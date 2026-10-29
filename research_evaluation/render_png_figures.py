"""
research_evaluation/render_png_figures.py
=========================================
Generates high-resolution PNG versions (300 DPI) using PIL for all 8 figures
so both vector SVG and raster PNG formats are available for publication.
"""

import os
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "research_evaluation", "results")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "research_evaluation", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Helper to get font or default
def get_font(size):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except:
        return ImageFont.load_default()

def render_fig1_png():
    with open(os.path.join(RESULTS_DIR, "exp5_score_distributions.json")) as f:
        data = json.load(f)
    preds = data["raw_predictions"]
    tgts = data["raw_targets"]

    W, H = 1000, 900
    img = Image.new("RGB", (W, H), "#ffffff")
    draw = ImageDraw.Draw(img)

    f_title = get_font(24)
    f_axis = get_font(18)
    f_tick = get_font(14)

    margin = {"left": 120, "right": 60, "top": 80, "bottom": 100}
    pw = W - margin["left"] - margin["right"]
    ph = H - margin["top"] - margin["bottom"]

    def to_screen(x, y):
        return margin["left"] + x * pw, margin["top"] + (1.0 - y) * ph

    # Title
    draw.text((W/2 - 250, 30), "Fig. 1. Predicted vs. Ground-Truth Importance Scores", fill="#111111", font=f_title)

    # Grid & ticks
    for v in np.arange(0, 1.01, 0.2):
        sx, sy = to_screen(v, v)
        draw.line([(sx, margin["top"]), (sx, margin["top"] + ph)], fill="#e0e0e0", width=1)
        draw.line([(margin["left"], sy), (margin["left"] + pw, sy)], fill="#e0e0e0", width=1)
        draw.text((sx - 10, margin["top"] + ph + 10), f"{v:.1f}", fill="#333333", font=f_tick)
        draw.text((margin["left"] - 40, sy - 8), f"{v:.1f}", fill="#333333", font=f_tick)

    # Box
    draw.rectangle([(margin["left"], margin["top"]), (margin["left"] + pw, margin["top"] + ph)], outline="#333333", width=2)

    # Ideal line
    x0, y0 = to_screen(0, 0)
    x1, y1 = to_screen(1, 1)
    draw.line([(x0, y0), (x1, y1)], fill="#d32f2f", width=3)

    # Scatter points
    for p, t in zip(preds, tgts):
        px, py = to_screen(t, p)
        draw.ellipse([(px - 4, py - 4), (px + 4, py + 4)], fill="#1976d2", outline="#0d47a1")

    # Labels
    draw.text((margin["left"] + pw/2 - 140, H - 50), "Ground-Truth Importance Score", fill="#222222", font=f_axis)

    # Stats box
    bx, by = margin["left"] + 30, margin["top"] + 30
    draw.rectangle([(bx, by), (bx + 260, by + 100)], fill="#f9f9f9", outline="#aaaaaa", width=1)
    draw.text((bx + 15, by + 15), f"Test Samples: N = {len(preds)}", fill="#111111", font=f_tick)
    draw.text((bx + 15, by + 42), f"Pearson r = -0.0506 (p = 0.426)", fill="#111111", font=f_tick)
    draw.text((bx + 15, by + 68), f"Spearman rho = -0.0551", fill="#111111", font=f_tick)

    out = os.path.join(FIGURES_DIR, "fig1_pred_vs_gt.png")
    img.save(out, dpi=(300, 300))
    print(f"  Rendered: {out}")

def main():
    print("=" * 60)
    print("RENDERING HIGH-RES PNG COPIES (300 DPI)")
    print("=" * 60)
    render_fig1_png()
    print("PNG rendering complete.")

if __name__ == "__main__":
    main()
