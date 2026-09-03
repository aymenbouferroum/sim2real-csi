"""Reproduce Figure 5: per-device aggregate fidelity at 20 dB in the laboratory.

Evaluates M1 AWGN and M_QTC consistency across 5 ESP32-C6 receivers with
per-device calibration. Uses 70/30 window-level splits with 5 seeds.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from mqtc.data import load_npz_csi
from mqtc.metrics import aggregate_fidelity_score
from mqtc.models import M1AWGN, MQTC

# --- Configuration ---

DATA_DIR = os.path.join(SCRIPT_DIR, "data", "laboratory", "constant_gaussian_20dB")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results", "per_device_20dB")

DEVICES = [
    "node-acoustic-02",
    "node-infrared-03",
    "node-magnetic-05",
    "node-optical-04",
    "node-thermal-01",
]

SEEDS = [42, 123, 456, 789, 1024]

MODELS = [
    ("M1_AWGN", M1AWGN),
    ("M_QTC", MQTC),
]

METRIC_KEYS = ("amplitude", "phase", "temporal", "spectral", "aggregate")


def _json_default(obj):
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def validate_single_seed(clean_csi, jammed_csi, seed, train_fraction=0.7):
    """Run within-scenario validation for one seed."""
    n_min = min(clean_csi.shape[0], jammed_csi.shape[0])
    clean_csi = clean_csi[:n_min]
    jammed_csi = jammed_csi[:n_min]

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_min)
    n_train = int(n_min * train_fraction)

    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    clean_train = clean_csi[train_idx]
    jammed_train = jammed_csi[train_idx]
    clean_test = clean_csi[test_idx]
    jammed_test = jammed_csi[test_idx]

    results = {}
    for model_name, model_cls in MODELS:
        model = model_cls(seed=seed)
        model.calibrate(clean_train, jammed_train)
        simulated_test = model.simulate(clean_test)

        scores = aggregate_fidelity_score(simulated_test, jammed_test)
        results[model_name] = {k: float(scores[k]) for k in METRIC_KEYS}

    return results


def main():
    print("=" * 70)
    print("Per-Device Fidelity Evaluation")
    print("Lab environment, constant Gaussian 20 dB")
    print(f"Devices: {len(DEVICES)}, Seeds: {SEEDS}")
    print(f"Models: {[name for name, _ in MODELS]}")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device_summaries = {}

    for device in DEVICES:
        print(f"\n{'='*70}")
        print(f"Device: {device}")
        print(f"{'='*70}")

        baseline_path = os.path.join(DATA_DIR, f"{device}_baseline.npz")
        jammed_path = os.path.join(DATA_DIR, f"{device}_jammed.npz")

        clean_csi = load_npz_csi(baseline_path)
        jammed_csi = load_npz_csi(jammed_path)
        print(f"  Clean windows:  {clean_csi.shape[0]}")
        print(f"  Jammed windows: {jammed_csi.shape[0]}")

        seed_results = []
        for seed in SEEDS:
            print(f"  Seed {seed}...", end=" ", flush=True)
            sr = validate_single_seed(clean_csi, jammed_csi, seed=seed)
            seed_results.append(sr)
            for model_name, _ in MODELS:
                agg = sr[model_name]["aggregate"]
                print(f"{model_name}={agg:.3f}", end="  ")
            print()

        # Aggregate across seeds
        model_names = [name for name, _ in MODELS]
        aggregated = {}
        for model_name in model_names:
            means = {}
            stds = {}
            for key in METRIC_KEYS:
                values = [sr[model_name][key] for sr in seed_results]
                means[key] = float(np.mean(values))
                stds[key] = float(np.std(values))
            aggregated[model_name] = {"mean": means, "std": stds}

        device_summaries[device] = aggregated

        out_path = os.path.join(OUTPUT_DIR, f"{device}.json")
        with open(out_path, "w") as f:
            json.dump(aggregated, f, indent=2, default=_json_default)
        print(f"  Saved: {out_path}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Per-device aggregate fidelity (mean across seeds)")
    print("Lower is better.")
    print("=" * 70)

    model_names = [name for name, _ in MODELS]

    header = f"{'Device':<22}"
    for model_name in model_names:
        header += f" | {model_name:>16}"
    print(header)
    print("-" * len(header))

    for device in DEVICES:
        row = f"{device:<22}"
        for model_name in model_names:
            m = device_summaries[device][model_name]["mean"]["aggregate"]
            s = device_summaries[device][model_name]["std"]["aggregate"]
            row += f" | {m:>7.3f} +/- {s:.3f}"
        print(row)

    print("-" * len(header))
    row = f"{'Mean across devices':<22}"
    for model_name in model_names:
        device_aggs = [
            device_summaries[d][model_name]["mean"]["aggregate"]
            for d in DEVICES
        ]
        row += f" | {np.mean(device_aggs):>7.3f} +/- {np.std(device_aggs):.3f}"
    print(row)

    # Per-metric breakdown
    print("\n" + "=" * 70)
    print("PER-METRIC BREAKDOWN (mean across 5 devices, mean across seeds)")
    print("=" * 70)

    for model_name in model_names:
        print(f"\n  {model_name}:")
        print(f"    {'Metric':<12} {'Mean':>8} {'Std':>8}")
        print(f"    {'-'*12} {'-'*8} {'-'*8}")
        for key in METRIC_KEYS:
            device_vals = [
                device_summaries[d][model_name]["mean"][key]
                for d in DEVICES
            ]
            print(f"    {key:<12} {np.mean(device_vals):>8.4f} {np.std(device_vals):>8.4f}")

    print("\n" + "=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
