"""Prints Table V (Section VI-E): external validation on public datasets.

Data: benchmarking/results/{wallhack_NLOS_BQ,signfi_antenna1,widar_crossroom}.json.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import bold_min, load_json, print_latex_table

BENCH_DIR = Path(__file__).resolve().parent.parent / "benchmarking" / "results"

# JSON key -> display label
METHOD_MAP = [
    ("MQTCFlexible",         r"M\textsubscript{QTC}"),
    ("M1AWGN",               "Gao et al."),
    ("AirFiAmplitudeNoise",  "AirFi"),
    ("StrohmayerScalingFlex", "Strohmayer et al."),
    ("SerbetciCombinedFlex", "Serbetci et al."),
]

# dataset filename -> column header
DATASET_MAP = [
    ("wallhack_NLOS_BQ.json", "Wallhack1.8k"),
    ("signfi_antenna1.json",  "SignFi"),
    ("widar_crossroom.json",  "Widar 3.0"),
]

HEADERS = ["Method"] + [name for _, name in DATASET_MAP]

CAPTION = (
    r"Amplitude Wasserstein distance on three external public datasets "
    r"(5 seeds, lower is better). Wallhack1.8k uses ESP32 (52 subcarriers), "
    r"SignFi and Widar use Intel 5300 (30 subcarriers)."
)
LABEL = "tab:external"


def main() -> None:
    """Print Table V: external validation."""
    datasets: list[dict] = []
    for fname, _ in DATASET_MAP:
        fpath = BENCH_DIR / fname
        if not fpath.exists():
            print(f"source data not found: {fpath}. Run `python reproduce_table.py`.")
            sys.exit(2)
        datasets.append(load_json(fpath)["results"])

    # collect raw amplitude values per dataset column
    raw_cols: list[list[float]] = []
    for ds in datasets:
        col: list[float] = []
        for json_key, _ in METHOD_MAP:
            col.append(ds[json_key]["mean"]["amplitude"])
        raw_cols.append(col)

    formatted_cols: list[list[str]] = []
    for col in raw_cols:
        formatted_cols.append(bold_min(col, fmt=".2f"))

    # build rows
    rows: list[list[str]] = []
    for i, (_, label) in enumerate(METHOD_MAP):
        row = [label]
        for fc in formatted_cols:
            row.append(fc[i])
        rows.append(row)

    print_latex_table(
        headers=HEADERS,
        rows=rows,
        caption=CAPTION,
        label=LABEL,
    )


if __name__ == "__main__":
    main()
