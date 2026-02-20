#!/usr/bin/env python3
"""
08_generate_report.py — Comprehensive HTML Results Report

Compiles all analysis outputs into a single, self-contained HTML report
with embedded figures, tables, and statistical results.

Outputs:
    results/report/sleep_network_reorganization_report.html
"""

import base64
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

warnings.filterwarnings("ignore")


def img_to_base64(img_path):
    """Convert an image file to base64 for embedding in HTML."""
    if not img_path.exists():
        return None
    with open(img_path, "rb") as f:
        data = f.read()
    ext = img_path.suffix.lstrip(".")
    mime = {"png": "image/png", "jpg": "image/jpeg", "pdf": "application/pdf"}.get(ext, "image/png")
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def load_json_safe(path):
    """Load JSON file, return empty dict on error."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def generate_report():
    """Generate the HTML report."""
    report_path = config.REPORT_DIR / "sleep_network_reorganization_report.html"

    # Load results
    qc_report = load_json_safe(config.QC_DIR / "data_validation_report.json")
    glm_log = load_json_safe(config.GLM_DIR / "glm_results_log.json")
    dfc_stats = load_json_safe(config.DFC_DIR / "stage_comparison_stats.json")
    thc_stats = load_json_safe(config.THALAMOCORTICAL_DIR / "anova_results.json")

    icc_results = load_json_safe(config.RELIABILITY_DIR / "icc_full_results.json")
    split_half = load_json_safe(config.RELIABILITY_DIR / "split_half_results.json")

    # Collect figures
    figures = {}
    fig_names = [
        "fig1_study_overview",
        "qc_stage_distribution", "qc_motion_summary",
        "glm_Wake_gt_N2_glass_brain", "glm_Wake_gt_N2_ortho",
        "glm_Wake_gt_N1_glass_brain", "glm_N1_gt_N2_glass_brain",
        "dfc_graph_metrics", "dfc_fc_matrices",
        "thalamocortical_radar", "thalamocortical_barplot", "thalamocortical_heatmap",

        "reliability_icc", "reliability_split_half",
        "fig7_summary_trajectory",
    ]
    for name in fig_names:
        png_path = config.FIGURES_DIR / f"{name}.png"
        if png_path.exists():
            figures[name] = img_to_base64(png_path)

    # Build HTML
    html_parts = []

    # --- Header ---
    html_parts.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sleep Network Reorganization — Analysis Report</title>
<style>
    body {{
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
        background-color: #fafafa;
        color: #333;
        line-height: 1.6;
    }}
    h1 {{
        color: #1565C0;
        border-bottom: 3px solid #1565C0;
        padding-bottom: 10px;
    }}
    h2 {{
        color: #2E7D32;
        border-bottom: 2px solid #E0E0E0;
        padding-bottom: 5px;
        margin-top: 40px;
    }}
    h3 {{
        color: #6A1B9A;
    }}
    .summary-box {{
        background: linear-gradient(135deg, #E3F2FD, #E8EAF6);
        border-left: 5px solid #1565C0;
        padding: 15px 20px;
        margin: 20px 0;
        border-radius: 0 8px 8px 0;
    }}
    .stat-box {{
        background-color: #F5F5F5;
        border: 1px solid #E0E0E0;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }}
    .warning {{
        background-color: #FFF3E0;
        border-left: 5px solid #FF9800;
        padding: 10px 15px;
        margin: 10px 0;
    }}
    table {{
        border-collapse: collapse;
        width: 100%;
        margin: 15px 0;
        background: white;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }}
    th {{
        background: linear-gradient(135deg, #1565C0, #1976D2);
        color: white;
        padding: 12px 15px;
        text-align: left;
    }}
    td {{
        padding: 10px 15px;
        border-bottom: 1px solid #E0E0E0;
    }}
    tr:nth-child(even) {{ background-color: #F5F5F5; }}
    tr:hover {{ background-color: #E3F2FD; }}
    img {{
        max-width: 100%;
        height: auto;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        margin: 15px 0;
    }}
    .figure-caption {{
        text-align: center;
        font-style: italic;
        color: #666;
        margin-top: 5px;
    }}
    .footer {{
        margin-top: 60px;
        padding-top: 20px;
        border-top: 2px solid #E0E0E0;
        text-align: center;
        color: #999;
        font-size: 0.9em;
    }}
    .metric-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 15px;
        margin: 15px 0;
    }}
    .metric-card {{
        background: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        text-align: center;
    }}
    .metric-value {{
        font-size: 2em;
        font-weight: bold;
        color: #1565C0;
    }}
    .metric-label {{
        color: #666;
        font-size: 0.9em;
    }}
</style>
</head>
<body>
<h1>🧠 Systematic Network Reorganization Across Sleep Transitions</h1>
<p><strong>Dataset:</strong> ds003768 — Simultaneous EEG and fMRI signals during sleep<br>
<strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
""")

    # --- Executive Summary ---
    summary = qc_report.get("summary", {})
    total_trs = summary.get("total_valid_trs", {})

    html_parts.append(f"""
<div class="summary-box">
<h2 style="margin-top:0; border:none;">Executive Summary</h2>
<p><strong>Research Question:</strong> Does large-scale brain network organization change
systematically during the transition from wakefulness to sleep (Wake → N1 → N2)?</p>
<div class="metric-grid">
    <div class="metric-card">
        <div class="metric-value">{summary.get('included_subjects', '—')}</div>
        <div class="metric-label">Included Subjects</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{total_trs.get('Total', '—')}</div>
        <div class="metric-label">Total Valid TRs</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{total_trs.get('Wake', '—')}</div>
        <div class="metric-label">Wake TRs</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{total_trs.get('N1', '—')}</div>
        <div class="metric-label">N1 TRs</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{total_trs.get('N2', '—')}</div>
        <div class="metric-label">N2 TRs</div>
    </div>
</div>
</div>
""")

    # --- QC Section ---
    html_parts.append("<h2>1. Data Quality & Validation</h2>")
    html_parts.append(f"""
<div class="stat-box">
<p><strong>FD threshold:</strong> {config.FD_THRESHOLD} mm (Power et al., 2014)<br>
<strong>Maximum censored fraction:</strong> {config.MAX_CENSORED_FRACTION:.0%}<br>
<strong>Excluded subjects:</strong> {summary.get('excluded_subject_ids', [])}</p>
</div>
""")

    for fig_name in ["fig1_study_overview", "qc_stage_distribution", "qc_motion_summary"]:
        if fig_name in figures:
            html_parts.append(f'<img src="{figures[fig_name]}" alt="{fig_name}">')
            html_parts.append(f'<p class="figure-caption">{fig_name.replace("_", " ").title()}</p>')

    # --- GLM Section ---
    html_parts.append("<h2>2. GLM Analysis</h2>")
    html_parts.append("""
<p>Sleep stages modeled as block regressors in a first-level GLM (nilearn).
Group-level analysis uses one-sample t-test (random effects).</p>
""")

    glm_subjects = glm_log.get("subjects", {})
    n_completed = sum(1 for s in glm_subjects.values() if s.get("status") == "completed")
    html_parts.append(f'<p><strong>{n_completed}</strong> subjects completed GLM analysis.</p>')

    for fig_name in ["glm_Wake_gt_N2_glass_brain", "glm_Wake_gt_N2_ortho",
                     "glm_Wake_gt_N1_glass_brain", "glm_N1_gt_N2_glass_brain"]:
        if fig_name in figures:
            html_parts.append(f'<img src="{figures[fig_name]}" alt="{fig_name}">')
            html_parts.append(f'<p class="figure-caption">{fig_name.replace("_", " ").replace("gt", ">").title()}</p>')

    # --- dFC Section ---
    html_parts.append("<h2>3. Dynamic Functional Connectivity</h2>")
    html_parts.append(f"""
<div class="stat-box">
<p><strong>Window size:</strong> {config.DFC_WINDOW_SIZE} TRs ({config.DFC_WINDOW_SIZE * config.TR:.1f}s)
(Leonardi & Van De Ville, 2015)<br>
<strong>Graph density threshold:</strong> {config.GRAPH_DENSITY_THRESHOLD}<br>
<strong>Modularity iterations:</strong> {config.MODULARITY_N_ITER} (Louvain algorithm)</p>
</div>
""")

    # dFC statistics
    for metric in ["global_efficiency", "modularity"]:
        if metric in dfc_stats:
            m_res = dfc_stats[metric]
            if "effect_sizes" in m_res:
                html_parts.append(f"<h3>{metric.replace('_', ' ').title()}</h3>")
                html_parts.append("<table><tr><th>Comparison</th><th>Cohen's d</th><th>Mean Diff</th></tr>")
                for pair, eff in m_res["effect_sizes"].items():
                    html_parts.append(
                        f"<tr><td>{pair.replace('_', ' ')}</td>"
                        f"<td>{eff['cohens_d']:.3f}</td>"
                        f"<td>{eff['mean_diff']:.4f}</td></tr>"
                    )
                html_parts.append("</table>")

    for fig_name in ["dfc_graph_metrics", "dfc_fc_matrices"]:
        if fig_name in figures:
            html_parts.append(f'<img src="{figures[fig_name]}" alt="{fig_name}">')

    # --- Thalamocortical Section ---
    html_parts.append("<h2>4. Thalamocortical Coupling</h2>")

    if "rm_anova_text" in thc_stats:
        html_parts.append(f"""
<div class="stat-box">
<h3>Repeated-Measures ANOVA</h3>
<pre>{thc_stats['rm_anova_text']}</pre>
</div>
""")

    for fig_name in ["thalamocortical_radar", "thalamocortical_barplot", "thalamocortical_heatmap"]:
        if fig_name in figures:
            html_parts.append(f'<img src="{figures[fig_name]}" alt="{fig_name}">')



    # --- Reliability Section ---
    html_parts.append("<h2>5. Test-Retest Reliability</h2>")

    if icc_results:
        html_parts.append("<h3>ICC Results</h3>")
        html_parts.append("<table><tr><th>Metric</th><th>ICC</th><th>95% CI</th><th>Interpretation</th></tr>")
        for metric, res in icc_results.items():
            if "icc_value" in res:
                html_parts.append(
                    f"<tr><td>{metric}</td>"
                    f"<td>{res['icc_value']:.3f}</td>"
                    f"<td>[{res['ci95_lower']:.3f}, {res['ci95_upper']:.3f}]</td>"
                    f"<td>{res['interpretation']}</td></tr>"
                )
        html_parts.append("</table>")

    if split_half and "mean_split_half_r" in split_half:
        html_parts.append(f"""
<div class="stat-box">
<h3>Split-Half Reliability</h3>
<p><strong>Mean Spearman-Brown corrected r:</strong> {split_half['mean_split_half_r']:.3f}
(95% CI: [{split_half.get('ci95_lower', 0):.3f}, {split_half.get('ci95_upper', 0):.3f}])<br>
<strong>Interpretation:</strong> {split_half.get('interpretation', 'N/A')}</p>
</div>
""")

    for fig_name in ["reliability_icc", "reliability_split_half"]:
        if fig_name in figures:
            html_parts.append(f'<img src="{figures[fig_name]}" alt="{fig_name}">')

    # --- Summary Figure ---
    html_parts.append("<h2>6. Integrated Summary</h2>")
    if "fig7_summary_trajectory" in figures:
        html_parts.append(f'<img src="{figures["fig7_summary_trajectory"]}" alt="Summary">')

    # --- Methods Summary ---
    html_parts.append("""
<h2>7. Methods Summary</h2>
<div class="stat-box">
<h3>Key Parameters (Literature-Backed)</h3>
<table>
<tr><th>Parameter</th><th>Value</th><th>Reference</th></tr>
<tr><td>FD censoring threshold</td><td>0.3 mm</td><td>Power et al., NeuroImage 2014</td></tr>
<tr><td>Confound regression</td><td>24-param motion + 5 aCompCor</td><td>Satterthwaite et al., 2013; Behzadi et al., 2007</td></tr>
<tr><td>Bandpass filter</td><td>0.01–0.1 Hz</td><td>Biswal et al., 1995</td></tr>
<tr><td>Sliding window size</td><td>44 TRs (92.4s)</td><td>Leonardi & Van De Ville, NeuroImage 2015</td></tr>
<tr><td>Graph density</td><td>15%</td><td>Rubinov & Sporns, NeuroImage 2010</td></tr>

<tr><td>Atlas</td><td>Schaefer 200-parcel 7-network</td><td>Schaefer et al., Cereb. Cortex 2018</td></tr>
<tr><td>Reliability</td><td>ICC(3,1)</td><td>Shrout & Fleiss, Psychol. Bull. 1979</td></tr>
</table>
</div>
""")

    # --- Footer ---
    html_parts.append(f"""
<div class="footer">
<p>Sleep Network Reorganization Analysis Pipeline<br>
Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
Dataset: ds003768 | Preprocessed with fMRIPrep 23.0.2</p>
</div>
</body>
</html>
""")

    # Write report
    report_html = "\n".join(html_parts)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_html)

    print(f"✓ Report generated: {report_path}")
    size_mb = report_path.stat().st_size / (1024 * 1024)
    print(f"  Size: {size_mb:.1f} MB")

    return report_path


def main():
    print("=" * 70)
    print("SLEEP NETWORK REORGANIZATION — HTML REPORT GENERATION")
    print("=" * 70)

    config.setup_directories()
    generate_report()

    print("\n✓ Report generation complete.")


if __name__ == "__main__":
    main()
