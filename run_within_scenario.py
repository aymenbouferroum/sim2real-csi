"""Reproduce Table II: within-scenario fidelity at 20 dB (controlled room).

Evaluates M1 AWGN, M2 PowerScaled, M3 Hybrid, and M_QTC against real
jammed CSI. Uses 70/30 random window-level splits with 5 seeds.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from mqtc.data import load_npz_csi
from mqtc.metrics import aggregate_fidelity_score
from mqtc.models import M1AWGN, M2PowerScaled, M3Hybrid, MQTC

# --- Configuration ---

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "controlled" / "constant_gaussian_20dB"
OUTPUT_DIR = PROJECT_ROOT / "results"

MODEL_CLASSES = [M1AWGN, M2PowerScaled, M3Hybrid, MQTC]  # paper order
SEEDS = [42, 123, 456, 789, 1024]
METRIC_KEYS = ("amplitude", "phase", "temporal", "spectral", "aggregate")


def _json_default(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    return x


def split_windows(n_windows: int, train_fraction: float = 0.7, seed: int = 42):
    """70/30 window-level split via seeded permutation."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_windows)
    n_train = int(n_windows * train_fraction)
    return perm[:n_train], perm[n_train:]


def evaluate_single_seed(clean_csi, jammed_csi, seed, train_fraction=0.7):
    """Run all four models for one train/test split."""
    n_min = min(clean_csi.shape[0], jammed_csi.shape[0])
    clean_csi = clean_csi[:n_min]
    jammed_csi = jammed_csi[:n_min]

    train_idx, test_idx = split_windows(n_min, train_fraction, seed)
    clean_train = clean_csi[train_idx]
    jammed_train = jammed_csi[train_idx]
    clean_test = clean_csi[test_idx]
    jammed_test = jammed_csi[test_idx]

    results = {}
    for model_cls in MODEL_CLASSES:
        model = model_cls(seed=seed)
        model.calibrate(clean_train, jammed_train)
        simulated_test = model.simulate(clean_test)
        scores = aggregate_fidelity_score(simulated_test, jammed_test)

        model_name = model.get_params()["model"]
        results[model_name] = {k: float(scores[k]) for k in METRIC_KEYS}

    return results


def aggregate_across_seeds(seed_results):
    """Mean and std across seed runs for each model and metric."""
    model_names = list(seed_results[0].keys())
    aggregated = {}

    for model_name in model_names:
        means = {}
        stds = {}
        for key in METRIC_KEYS:
            values = [sr[model_name][key] for sr in seed_results]
            means[key] = float(np.mean(values))
            stds[key] = float(np.std(values))
        aggregated[model_name] = {"mean": means, "std": stds}

    return aggregated


def main():
    baseline_path = DATA_DIR / "baseline.npz"
    jammed_path = DATA_DIR / "jammed.npz"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading baseline: {baseline_path}")
    clean_csi = load_npz_csi(str(baseline_path))
    print(f"  Shape: {clean_csi.shape} ({clean_csi.shape[0]} windows)")

    print(f"Loading jammed:   {jammed_path}")
    jammed_csi = load_npz_csi(str(jammed_path))
    print(f"  Shape: {jammed_csi.shape} ({jammed_csi.shape[0]} windows)")

    n_min = min(clean_csi.shape[0], jammed_csi.shape[0])
    print(f"Using {n_min} windows (truncated to shorter array)")
    print()

    # Run evaluation across all seeds
    seed_results = []
    for i, seed in enumerate(SEEDS):
        t0 = time.time()
        print(f"Seed {seed} ({i + 1}/{len(SEEDS)}) ... ", end="", flush=True)
        sr = evaluate_single_seed(clean_csi, jammed_csi, seed=seed)
        elapsed = time.time() - t0
        print(f"done ({elapsed:.1f}s)")

        for model_name, scores in sr.items():
            agg = scores["aggregate"]
            print(f"  {model_name:20s}  aggregate={agg:.4f}")
        print()

        seed_results.append(sr)

    # Aggregate mean and std across seeds
    aggregated = aggregate_across_seeds(seed_results)

    # Print summary table
    print("=" * 72)
    print("Table II: Within-scenario fidelity at 20 dB (controlled room)")
    print("=" * 72)
    header = f"{'Model':20s}"
    for key in METRIC_KEYS:
        header += f"  {key:>12s}"
    print(header)
    print("-" * 72)

    for model_name in aggregated:
        row = f"{model_name:20s}"
        for key in METRIC_KEYS:
            m = aggregated[model_name]["mean"][key]
            s = aggregated[model_name]["std"][key]
            row += f"  {m:5.3f}+/-{s:.3f}"
        print(row)
    print("=" * 72)

    # Save results
    output_path = OUTPUT_DIR / "controlled_20dB.json"
    with open(output_path, "w") as f:
        json.dump(aggregated, f, indent=2, default=_json_default)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
