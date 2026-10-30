"""
Generates a publication-grade IEEE/ACM Research Evaluation PDF from results_report.md
and embeds all 8 vector SVG figures with professional academic typography and layout.
"""

import os
import re
import sys
import base64
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
FIGURES_DIR = BASE_DIR / "figures"
OUTPUT_HTML = BASE_DIR / "publication_report.html"
OUTPUT_PDF = BASE_DIR / "Video_Summarization_Research_Evaluation_Report.pdf"

# Find Chrome or Edge
CHROME_PATH = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
EDGE_PATH = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")

def get_browser_exe():
    if CHROME_PATH.exists():
        return str(CHROME_PATH)
    if EDGE_PATH.exists():
        return str(EDGE_PATH)
    raise FileNotFoundError("Neither Google Chrome nor Microsoft Edge was found on the system.")

def read_svg_as_base64(svg_path: Path) -> str:
    if not svg_path.exists():
        return ""
    data = svg_path.read_bytes()
    b64 = base64.b64encode(data).decode('utf-8')
    return f"data:image/svg+xml;base64,{b64}"

def convert_markdown_tables(text: str) -> str:
    """Converts markdown tables to clean HTML tables with IEEE publication styling."""
    lines = text.split("\n")
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("|") and line.strip().endswith("|"):
            table_lines = [line]
            i += 1
            while i < len(lines) and lines[i].strip().startswith("|") and lines[i].strip().endswith("|"):
                table_lines.append(lines[i])
                i += 1
            
            # Parse table
            if len(table_lines) >= 3 and "---" in table_lines[1]:
                headers = [c.strip() for c in table_lines[0].strip().strip("|").split("|")]
                rows = []
                for row_line in table_lines[2:]:
                    cols = [c.strip() for c in row_line.strip().strip("|").split("|")]
                    rows.append(cols)
                
                html_table = ['<div class="table-container">', '<table class="ieee-table">', '<thead>', '<tr>']
                for h in headers:
                    # Clean markdown formatting inside headers
                    h_clean = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', h)
                    html_table.append(f'  <th>{h_clean}</th>')
                html_table.extend(['</tr>', '</thead>', '<tbody>'])
                for row in rows:
                    html_table.append('<tr>')
                    for c in row:
                        c_clean = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', c)
                        c_clean = re.sub(r'\*(.*?)\*', r'<em>\1</em>', c_clean)
                        c_clean = re.sub(r'`(.*?)`', r'<code>\1</code>', c_clean)
                        c_clean = re.sub(r'\$(.*?)\$', r'<i>\1</i>', c_clean)
                        html_table.append(f'  <td>{c_clean}</td>')
                    html_table.append('</tr>')
                html_table.extend(['</tbody>', '</table>', '</div>'])
                new_lines.append("\n".join(html_table))
            else:
                new_lines.extend(table_lines)
        else:
            new_lines.append(line)
            i += 1
    return "\n".join(new_lines)

def build_html_document():
    raw_md = (BASE_DIR / "results_report.md").read_text(encoding="utf-8")

    # Load SVG Figures as Base64 Data URIs
    svgs = {
        "fig1": read_svg_as_base64(FIGURES_DIR / "fig1_pred_vs_gt.svg"),
        "fig2": read_svg_as_base64(FIGURES_DIR / "fig2_score_distributions.svg"),
        "fig3": read_svg_as_base64(FIGURES_DIR / "fig3_ablation.svg"),
        "fig4": read_svg_as_base64(FIGURES_DIR / "fig4_cross_split.svg"),
        "fig5": read_svg_as_base64(FIGURES_DIR / "fig5_threshold_sensitivity.svg"),
        "fig6": read_svg_as_base64(FIGURES_DIR / "fig6_per_video.svg"),
        "fig7": read_svg_as_base64(FIGURES_DIR / "fig7_param_distribution.svg"),
        "fig8": read_svg_as_base64(FIGURES_DIR / "fig8_residuals.svg"),
    }

    # Convert tables
    processed = convert_markdown_tables(raw_md)

    # Process headers
    processed = re.sub(r'^# (.*?)$', r'<h1 class="paper-title">\1</h1>', processed, flags=re.M)
    processed = re.sub(r'^## (.*?)$', r'<h2 class="section-title">\1</h2>', processed, flags=re.M)
    processed = re.sub(r'^### (TABLE .*?)$', r'<div class="table-caption">\1</div>', processed, flags=re.M)
    processed = re.sub(r'^### (.*?)$', r'<h3 class="subsection-title">\1</h3>', processed, flags=re.M)

    # Process bold, italics, code, math
    processed = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', processed)
    processed = re.sub(r'\*(.*?)\*', r'<em>\1</em>', processed)
    processed = re.sub(r'`(.*?)`', r'<code>\1</code>', processed)
    processed = re.sub(r'\$\$(.*?)\$\$', r'<div class="equation"><em>\1</em></div>', processed)
    processed = re.sub(r'\$(.*?)\$', r'<span class="inline-math"><em>\1</em></span>', processed)
    processed = re.sub(r'^> (.*?)$', r'<blockquote class="ieee-quote">\1</blockquote>', processed, flags=re.M)

    # Convert paragraphs and figure placeholders
    def create_figure_block(fig_key, fig_num, title, description):
        img_src = svgs.get(fig_key, "")
        return f"""
        <div class="figure-card">
          <div class="figure-img-wrap">
            <img src="{img_src}" alt="Figure {fig_num}" class="vector-figure" />
          </div>
          <div class="figure-caption">
            <strong>Figure {fig_num}:</strong> <em>{title}</em> &mdash; {description}
          </div>
        </div>
        """

    # Insert figures into corresponding sections
    fig1_html = create_figure_block("fig1", 1, "Predicted vs. Ground-Truth Saliency Correlation", 
                                    "Scatter plot of 249 test segments across Checkpoints A and B. Blue line denotes y=x ideal calibration. Clustering around 0.86-0.88 confirms narrow dynamic variance.")
    fig2_html = create_figure_block("fig2", 2, "Score Distribution Density & Cumulative Dispersion", 
                                    "Kernel density estimation (KDE) and empirical cumulative distribution function (ECDF) contrasting ground truth against model outputs.")
    fig3_html = create_figure_block("fig3", 3, "Multimodal Feature Ablation Analysis", 
                                    "Mean Squared Error (MSE) degradation across Full Multimodal, Visual-Only, Text-Only, and Zero-Input configurations.")
    fig4_html = create_figure_block("fig4", 4, "Cross-Split Generalization Profile", 
                                    "Training, validation, and test set performance showing overfitting to synthetic sequence artifacts.")
    fig5_html = create_figure_block("fig5", 5, "Precision-Recall and Binary F1 Sensitivity", 
                                    "Metric response across classification thresholds theta in [0.1, 0.9], highlighting the sharp drop above theta > 0.85.")
    fig6_html = create_figure_block("fig6", 6, "Per-Video Rank Correlation Dispersion", 
                                    "Kendall's Tau and Spearman's Rho per test video demonstrating variance across SumMe and TVSum domains.")
    fig7_html = create_figure_block("fig7", 7, "Model Architecture Parameter Distribution", 
                                    "Proportional distribution of the 1,004,801 parameters across visual projection, text projection, fusion, BiLSTM, and saliency head.")
    fig8_html = create_figure_block("fig8", 8, "Prediction Residual Distribution and Error Normalcy", 
                                    "Residual distribution (y_hat - y) exhibiting near-Gaussian error centered at -0.008 with standard deviation 0.085.")

    # Substitute figure references with actual rendered HTML cards
    # Figure 1 & 2 in Section 4
    processed = processed.replace(
        "As illustrated in Figure 1",
        f"{fig1_html}\n{fig2_html}\nAs illustrated in Figure 1"
    )
    # Figure 3 in Section 5
    processed = processed.replace(
        "Results are presented in Table III and Figure 3",
        f"Results are presented in Table III, Figure 3, and Figure 5.\n{fig3_html}\n{fig5_html}"
    )
    # Figure 4 & 6 in Section 6
    processed = processed.replace(
        "evaluated across all three partitions (Table IV and Figure 4).",
        f"evaluated across all three partitions (Table IV, Figure 4, and Figure 6).\n{fig4_html}\n{fig6_html}"
    )
    # Figure 7 & 8 in Section 7
    processed = processed.replace(
        "As detailed in [`fig7_param_distribution.svg`]",
        f"{fig7_html}\n{fig8_html}\nAs detailed in Table V"
    )

    # Wrap in clean HTML template with IEEE publication styling
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>A Multimodal Framework for Long-Format Video Summarization - Research Evaluation Report</title>
<style>
  @page {{
    size: A4 portrait;
    margin: 18mm 14mm 20mm 14mm;
    @bottom-center {{
      content: counter(page);
      font-size: 9pt;
      font-family: 'Times New Roman', Times, serif;
      color: #666;
    }}
  }}

  * {{
    box-sizing: border-box;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }}

  body {{
    font-family: 'Times New Roman', Times, Georgia, serif;
    font-size: 10pt;
    line-height: 1.45;
    color: #111;
    background: #fff;
    margin: 0;
    padding: 0;
  }}

  /* HEADER & METADATA */
  .ieee-header {{
    text-align: center;
    border-bottom: 2px solid #222;
    padding-bottom: 12px;
    margin-bottom: 16px;
  }}

  .paper-title {{
    font-size: 19pt;
    font-weight: bold;
    line-height: 1.25;
    margin: 0 0 8px 0;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #0b1a30;
  }}

  .paper-subtitle {{
    font-size: 11pt;
    font-style: italic;
    color: #333;
    margin-bottom: 10px;
  }}

  .author-meta {{
    font-size: 9pt;
    color: #444;
    margin-bottom: 6px;
  }}

  .doc-badge-row {{
    display: flex;
    justify-content: center;
    gap: 15px;
    font-size: 8.5pt;
    font-family: 'Segoe UI', Arial, sans-serif;
    color: #2b4c7e;
    font-weight: 600;
    margin-top: 6px;
  }}

  /* SECTIONS */
  .section-title {{
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 12pt;
    font-weight: bold;
    text-transform: uppercase;
    color: #0d233a;
    border-bottom: 1px solid #0d233a;
    padding-bottom: 3px;
    margin-top: 20px;
    margin-bottom: 10px;
    page-break-after: avoid;
  }}

  .subsection-title {{
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 10.5pt;
    font-weight: bold;
    color: #1a365d;
    margin-top: 14px;
    margin-bottom: 6px;
    page-break-after: avoid;
  }}

  p {{
    margin: 0 0 8px 0;
    text-align: justify;
  }}

  code {{
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 8.5pt;
    background: #f0f3f8;
    padding: 1px 4px;
    border-radius: 3px;
    border: 1px solid #d4dce8;
  }}

  /* TABLES */
  .table-container {{
    margin: 12px 0 16px 0;
    page-break-inside: avoid;
    width: 100%;
  }}

  .table-caption {{
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 9pt;
    font-weight: bold;
    text-align: center;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
    color: #1a202c;
  }}

  table.ieee-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 8.5pt;
    line-height: 1.35;
    margin: 0 auto;
    border-top: 2px solid #111;
    border-bottom: 2px solid #111;
  }}

  table.ieee-table th {{
    border-top: 1px solid #111;
    border-bottom: 1px solid #111;
    padding: 6px 7px;
    text-align: left;
    background-color: #f7fafc;
    font-weight: bold;
    color: #0f172a;
  }}

  table.ieee-table td {{
    padding: 5px 7px;
    border-bottom: 1px solid #e2e8f0;
  }}

  table.ieee-table tr:nth-child(even) td {{
    background-color: #f8fafc;
  }}

  /* FIGURES */
  .figure-card {{
    margin: 14px 0 18px 0;
    padding: 8px;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    background: #ffffff;
    page-break-inside: avoid;
    text-align: center;
  }}

  .figure-img-wrap {{
    width: 100%;
    display: flex;
    justify-content: center;
    background: #ffffff;
    padding: 4px;
  }}

  .vector-figure {{
    max-width: 100%;
    max-height: 290px;
    width: auto;
    height: auto;
    display: block;
    margin: 0 auto;
  }}

  .figure-caption {{
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 8pt;
    line-height: 1.3;
    color: #334155;
    margin-top: 6px;
    padding-top: 4px;
    border-top: 1px solid #e2e8f0;
    text-align: justify;
  }}

  /* CALLOUTS & BLOCKQUOTES */
  .ieee-quote {{
    margin: 10px 0;
    padding: 8px 12px;
    background: #f1f5f9;
    border-left: 3px solid #2563eb;
    font-size: 9pt;
    font-style: italic;
    color: #1e293b;
    page-break-inside: avoid;
  }}

  .equation {{
    text-align: center;
    font-family: 'Times New Roman', serif;
    font-size: 10.5pt;
    margin: 8px 0;
    padding: 4px;
    background: #fdfdfe;
  }}

  ul, ol {{
    margin: 4px 0 10px 20px;
    padding: 0;
  }}

  li {{
    margin-bottom: 4px;
    text-align: justify;
  }}
</style>
</head>
<body>

<div class="ieee-header">
  <div class="paper-title">A Multimodal Framework for Long-Format Video Summarization</div>
  <div class="paper-subtitle">Visual Saliency Estimation via Bidirectional LSTM Feature Fusion and Hierarchical Language Model Synthesis</div>
  <div class="author-meta">
    <strong>Senior Machine Learning & Computer Vision Research Team</strong> &bull; Advanced AI Systems &bull; IEEE Empirical Evaluation Report
  </div>
  <div class="doc-badge-row">
    <span>Date: October 2026</span>
    <span>&bull;</span>
    <span>Benchmark: SumMe & TVSum</span>
    <span>&bull;</span>
    <span>Hardware: AMD Ryzen 5 CPU Execution</span>
    <span>&bull;</span>
    <span>Status: Complete Empirical Audit</span>
  </div>
</div>

<div class="content">
{processed}
</div>

</body>
</html>
"""
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"Generated HTML report at: {OUTPUT_HTML} ({len(html)} bytes)")

def convert_html_to_pdf():
    browser = get_browser_exe()
    print(f"Rendering PDF using: {browser}")

    cmd = [
        browser,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={str(OUTPUT_PDF)}",
        str(OUTPUT_HTML)
    ]

    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    if OUTPUT_PDF.exists():
        size_kb = OUTPUT_PDF.stat().st_size / 1024
        print(f"SUCCESS: Publication-Ready PDF created at: {OUTPUT_PDF} ({size_kb:.1f} KB)")
    else:
        print(f"ERROR: PDF was not generated. Stderr: {res.stderr}")

if __name__ == "__main__":
    build_html_document()
    convert_html_to_pdf()
