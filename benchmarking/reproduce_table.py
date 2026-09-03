#!/usr/bin/env python3
"""Reproduce Table: M_QTC vs published methods, amplitude Wasserstein distance.

Runs 5 methods x 3 datasets x 5 seeds; prints plain-text and LaTeX tables.
Datasets must already be present under data/ (see README).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mqtc.models.m1_awgn import M1AWGN
from mqtc_flexible import MQTCFlexible
from competitor_methods import AirFiAmplitudeNoise
from external_models_flexible import StrohmayerScalingFlex, SerbetciCombinedFlex
from external_benchmark import run_paired_benchmark

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DATA_DIR = Path(__file__).resolve().parent / "data"

MODELS = [
    MQTCFlexible,
    M1AWGN,
    AirFiAmplitudeNoise,
    StrohmayerScalingFlex,
    SerbetciCombinedFlex,
]

MODEL_NAMES = {
    "MQTCFlexible": "M_QTC (ours)",
    "M1AWGN": "Gao et al.",
    "AirFiAmplitudeNoise": "AirFi (Wang et al.)",
    "StrohmayerScalingFlex": "Strohmayer et al.",
    "SerbetciCombinedFlex": "Serbetci et al.",
}

CONFIGS = [
    ("wallhack_NLOS_BQ", "Wallhack1.8k"),
    ("signfi_antenna1", "SignFi"),
    ("widar_crossroom", "Widar 3.0"),
]


# --- Data loading ------------------------------------------------------------

def load_wallhack():
    from loaders_external import load_wallhack_pair
    root = str(DATA_DIR / "wallhack" / "wallhack1.8k")
    return load_wallhack_pair(root, domain="NLOS/BQ", modified_prefixes=("w", "ww"))


def load_signfi():
    from loaders_external import load_signfi_pair
    sig = DATA_DIR / "signfi"
    return load_signfi_pair(
        str(sig / "dataset_lab_276_dl.mat"),
        str(sig / "dataset_home_276.mat"),
        antenna=1, max_windows=3000,
    )


def load_widar():
    from loaders_external import load_widar_crossroom
    return load_widar_crossroom(
        str(DATA_DIR / "widar"),
        room1="20181115", room2="20181117",
        max_windows=3000,
    )


LOADERS = {
    "wallhack_NLOS_BQ": ("Wallhack1.8k NLOS/BQ (presence)", load_wallhack),
    "signfi_antenna1": ("SignFi antenna 1 (lab->home)", load_signfi),
    "widar_crossroom": ("Widar 3.0 (Room1->Room2)", load_widar),
}


# --- Run benchmark -----------------------------------------------------------

def run_all():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for key, (label, loader) in LOADERS.items():
        print(f"\n=== {label} ===")
        clean, modified = loader()
        results = run_paired_benchmark(clean, modified, MODELS, verbose=True)
        out = RESULTS_DIR / f"{key}.json"
        with open(out, "w") as f:
            json.dump({"results": results}, f, indent=2)
        print(f"  saved -> {out}")


# --- Print table -------------------------------------------------------------

def print_table():
    print("\n" + "=" * 70)
    print("PAPER TABLE: Amplitude Wasserstein distance (5 seeds, lower = better)")
    print("=" * 70)

    header = f"{'Method':24s}"
    for _, col_label in CONFIGS:
        header += f" | {col_label:>14s}"
    print(header)
    print("-" * len(header))

    for model_cls in MODELS:
        name = MODEL_NAMES[model_cls.__name__]
        row = f"{name:24s}"
        for key, _ in CONFIGS:
            f = RESULTS_DIR / f"{key}.json"
            d = json.load(open(f))["results"]
            a = d[model_cls.__name__]["mean"]["amplitude"]
            row += f" | {a:14.2f}"
        print(row)

    print()
    print("LaTeX:")
    print()
    print(r"\begin{table}[t]")
    print(r"  \centering")
    print(r"  \caption{Amplitude Wasserstein distance (lower = better) for")
    print(r"  M\textsubscript{QTC} vs.\ published CSI augmentation methods on")
    print(r"  three external datasets (\NumSeeds{} seeds).}")
    print(r"  \label{tab:external}")
    print(r"  \small")
    print(r"  \begin{tabular}{lrrr}")
    print(r"    \toprule")
    print(r"    & Wallhack1.8k & SignFi & Widar~3.0 \\")
    print(r"    Method & (presence) & (lab$\to$home) & (room change) \\")
    print(r"    \midrule")

    for model_cls in MODELS:
        vals = []
        for key, _ in CONFIGS:
            f = RESULTS_DIR / f"{key}.json"
            d = json.load(open(f))["results"]
            vals.append(d[model_cls.__name__]["mean"]["amplitude"])
        texname = MODEL_NAMES[model_cls.__name__]
        if model_cls.__name__ == "MQTCFlexible":
            cells = " & ".join(f"\\textbf{{{v:.2f}}}" for v in vals)
            print(f"    M\\textsubscript{{QTC}} & {cells} \\\\")
        else:
            cells = " & ".join(f"{v:.2f}" for v in vals)
            print(f"    {texname} & {cells} \\\\")

    print(r"    \bottomrule")
    print(r"  \end{tabular}")
    print(r"\end{table}")


# --- Main --------------------------------------------------------------------

def main():
    print("Running benchmark (5 methods x 3 datasets x 5 seeds)")
    run_all()
    print_table()


if __name__ == "__main__":
    main()
