"""Prints Table III (Section VI-C): ablation of M_QTC components.

Data: results/ablation_20dB.json, results/ablation_lab_20dB.json.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import bold_min, load_json, print_latex_table

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# JSON key -> display label
VARIANT_MAP = [
    ("Q",   "Q-only"),
    ("QT",  "QT"),
    ("QC",  "QC"),
    ("QTC", "QTC"),
]

DIM_METRICS = ["amplitude", "temporal", "spectral"]

HEADERS = ["Variant", "Ampl.", "Temp.", "Spect.", "Aggr. (ctrl)", "Aggr. (lab)"]

CAPTION = (
    r"Ablation of M\textsubscript{QTC} at 20~dB (mean$\pm$std, 5 seeds). "
    r"Q = quantile mapping, T = AR(1), C = copula. Per-dimension scores: "
    r"controlled room; aggregates: both environments."
)
LABEL = "tab:ablation"


def main() -> None:
    """Print Table III: ablation study."""
    ctrl_path = RESULTS_DIR / "ablation_20dB.json"
    lab_path = RESULTS_DIR / "ablation_lab_20dB.json"
    for fpath in (ctrl_path, lab_path):
        if not fpath.exists():
            print(f"source data not found: {fpath}. Run the ablation benchmark.")
            sys.exit(2)

    ctrl = load_json(ctrl_path)
    lab = load_json(lab_path)

    # collect raw values per column for bold-min highlighting
    raw_dim: dict[str, list[float]] = {m: [] for m in DIM_METRICS}
    raw_ctrl_agg: list[float] = []
    raw_lab_agg: list[float] = []

    for json_key, _ in VARIANT_MAP:
        ctrl_means = ctrl[json_key]["mean"]
        for m in DIM_METRICS:
            raw_dim[m].append(ctrl_means[m])
        raw_ctrl_agg.append(ctrl_means["aggregate"])
        raw_lab_agg.append(lab[json_key]["mean"]["aggregate"])

    formatted_dim: dict[str, list[str]] = {}
    for m in DIM_METRICS:
        formatted_dim[m] = bold_min(raw_dim[m], fmt=".2f")

    fmt_ctrl_agg = bold_min(raw_ctrl_agg, fmt=".2f")
    fmt_lab_agg = bold_min(raw_lab_agg, fmt=".2f")

    # build rows
    rows: list[list[str]] = []
    for i, (json_key, label) in enumerate(VARIANT_MAP):
        row = [label]
        for m in DIM_METRICS:
            row.append(formatted_dim[m][i])
        ctrl_std = ctrl[json_key]["std"]["aggregate"]
        row.append(f"{fmt_ctrl_agg[i]}$\\pm${ctrl_std:.2f}")
        lab_std = lab[json_key]["std"]["aggregate"]
        row.append(f"{fmt_lab_agg[i]}$\\pm${lab_std:.2f}")
        rows.append(row)

    print_latex_table(
        headers=HEADERS,
        rows=rows,
        caption=CAPTION,
        label=LABEL,
        tabcolsep="4pt",
    )


if __name__ == "__main__":
    main()
