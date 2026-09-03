"""Generic paired-CSI fidelity benchmark, reusable across datasets.

Multi-seed 70/30 within-scenario protocol: calibrate on train, simulate on test, score.
"""

import json
from pathlib import Path

import numpy as np

from mqtc.metrics.aggregate import aggregate_fidelity_score

DEFAULT_SEEDS = (42, 123, 456, 789, 1024)
METRIC_KEYS = ("amplitude", "phase", "temporal", "spectral", "aggregate")


def split_windows(n: int, frac: float = 0.7, seed: int = 42):
    """Random 70/30 train/test split over window indices."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(n * frac)
    return perm[:n_train], perm[n_train:]


def run_paired_benchmark(
    clean: np.ndarray,
    modified: np.ndarray,
    models: list,
    seeds=DEFAULT_SEEDS,
    train_frac: float = 0.7,
    verbose: bool = True,
):
    """Run the multi-seed fidelity benchmark on one paired CSI dataset."""
    n_min = min(clean.shape[0], modified.shape[0])
    clean, modified = clean[:n_min], modified[:n_min]

    if verbose:
        print(f"  paired windows: {n_min} | shape {clean.shape} | "
              f"subcarriers {clean.shape[-2]}")

    all_results = {}
    for model_cls in models:
        model_name = model_cls.__name__
        seed_results = {k: [] for k in METRIC_KEYS}

        for seed in seeds:
            train_idx, test_idx = split_windows(n_min, frac=train_frac, seed=seed)
            model = model_cls(seed=seed)
            model.calibrate(clean[train_idx], modified[train_idx])
            simulated = model.simulate(clean[test_idx])
            scores = aggregate_fidelity_score(simulated, modified[test_idx])
            for k in METRIC_KEYS:
                seed_results[k].append(float(scores[k]))

        all_results[model_name] = {
            "mean": {k: float(np.mean(seed_results[k])) for k in METRIC_KEYS},
            "std": {k: float(np.std(seed_results[k])) for k in METRIC_KEYS},
        }

        if verbose:
            m = all_results[model_name]["mean"]
            s = all_results[model_name]["std"]
            print(f"  {model_name:28s} | Ampl {m['amplitude']:.3f}+/-{s['amplitude']:.3f} | "
                  f"Phase {m['phase']:.3f} | Temp {m['temporal']:.3f} | "
                  f"Spect {m['spectral']:.2f} | Aggr {m['aggregate']:.3f}")

    return all_results


def save_results(results: dict, out_path, meta: dict | None = None) -> None:
    """Write benchmark results to JSON."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta or {}, "results": results}
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  saved -> {out_path}")
