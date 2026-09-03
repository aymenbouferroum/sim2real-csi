"""Reproduce Table III: M_QTC component ablation at 20 dB (controlled room).

Tests Q-only, QT, QC, and full QTC variants to isolate each component's
contribution. All variants share identical calibration; the mode flag
controls which stages are applied during simulation.
"""

from __future__ import annotations

import json
import time
from enum import Enum
from pathlib import Path

import numpy as np
from scipy.signal import lfilter
from statsmodels.tsa.stattools import acf

from mqtc.data import load_npz_csi
from mqtc.metrics import aggregate_fidelity_score
from mqtc.models.base import SimulationModel

# --- Configuration ---

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "controlled" / "constant_gaussian_20dB"
OUTPUT_DIR = PROJECT_ROOT / "results"

N_QUANTILES = 1000  # grid points for per-subcarrier quantile mapping
SEEDS = [42, 123, 456, 789, 1024]
METRIC_KEYS = ("amplitude", "phase", "temporal", "spectral", "aggregate")


class AblationMode(Enum):
    """Which M_QTC stages to apply during simulation."""
    Q = "Q"
    QT = "QT"
    QC = "QC"
    QTC = "QTC"


ABLATION_VARIANTS = [AblationMode.Q, AblationMode.QT, AblationMode.QC, AblationMode.QTC]


class MQTCAblation(SimulationModel):
    """M_QTC with selectable component stages for ablation."""

    def __init__(self, mode: AblationMode, seed: int = 42):
        self.mode = mode
        self.seed = seed
        self._clean_quantiles: np.ndarray | None = None
        self._jammed_quantiles: np.ndarray | None = None
        self._phi: float = 0.0
        self._residual_std: np.ndarray | None = None
        self._target_cholesky: np.ndarray | None = None
        self._phase_shift: np.ndarray | None = None

    def calibrate(self, clean_csi: np.ndarray, jammed_csi: np.ndarray) -> None:
        """Fit all three stages regardless of ablation mode."""
        clean_c = (clean_csi[..., 0] + 1j * clean_csi[..., 1]).reshape(-1, 52)
        jammed_c = (jammed_csi[..., 0] + 1j * jammed_csi[..., 1]).reshape(-1, 52)

        clean_amp = np.abs(clean_c)
        jammed_amp = np.abs(jammed_c)
        clean_phase = np.angle(clean_c)
        jammed_phase = np.angle(jammed_c)

        # Stage 1: per-subcarrier quantile mapping tables
        quantile_grid = np.linspace(0, 1, N_QUANTILES)
        self._clean_quantiles = np.empty((52, N_QUANTILES))
        self._jammed_quantiles = np.empty((52, N_QUANTILES))
        for k in range(52):
            self._clean_quantiles[k] = np.quantile(clean_amp[:, k], quantile_grid)
            self._jammed_quantiles[k] = np.quantile(jammed_amp[:, k], quantile_grid)

        # Stage 2: AR(1) coefficient from quantile-mapping residuals
        n_min = min(clean_amp.shape[0], jammed_amp.shape[0])
        mapped_clean = self._apply_quantile_map(clean_amp[:n_min])
        residuals = jammed_amp[:n_min] - mapped_clean

        residual_std = np.std(residuals, axis=0)
        self._residual_std = np.maximum(residual_std, 1e-10)

        lag1_acfs = []
        for k in range(52):
            series = residuals[:, k]
            if np.std(series) < 1e-10:
                continue
            acf_vals = acf(series, nlags=1, fft=True)
            lag1_acfs.append(acf_vals[1])
        self._phi = (
            float(np.clip(np.median(lag1_acfs), 0.0, 0.99))
            if lag1_acfs
            else 0.0
        )

        # Stage 3: target correlation matrix from jammed amplitudes
        target_corr = np.corrcoef(jammed_amp.T)
        try:
            self._target_cholesky = np.linalg.cholesky(
                target_corr + 1e-6 * np.eye(52)
            )
        except np.linalg.LinAlgError:
            self._target_cholesky = None

        # Phase: mean circular shift per subcarrier
        n_phase = min(clean_phase.shape[0], jammed_phase.shape[0])
        phase_diff = jammed_phase[:n_phase] - clean_phase[:n_phase]
        self._phase_shift = np.angle(np.mean(np.exp(1j * phase_diff), axis=0))

    def simulate(self, clean_csi: np.ndarray) -> np.ndarray:
        """Simulate using only the selected ablation stages."""
        if (
            self._clean_quantiles is None
            or self._jammed_quantiles is None
            or self._residual_std is None
            or self._phase_shift is None
        ):
            raise RuntimeError("Model not calibrated. Call calibrate() first.")

        rng = np.random.default_rng(self.seed)
        complex_csi = clean_csi[..., 0] + 1j * clean_csi[..., 1]
        orig_shape = complex_csi.shape

        clean_amp = np.abs(complex_csi)
        clean_phase = np.angle(complex_csi)

        flat_amp = clean_amp.reshape(-1, 52)
        n_samples = flat_amp.shape[0]

        # Stage 1 (always): quantile mapping
        mapped_amp = self._apply_quantile_map(flat_amp)

        if self.mode == AblationMode.Q:
            result_amp = mapped_amp

        elif self.mode == AblationMode.QT:
            # AR(1) on independent per-subcarrier scores, then Iman-Conover
            phi = self._phi
            innovation_scale = np.sqrt(max(1.0 - phi ** 2, 0.0))

            z_indep = rng.normal(0, 1, (n_samples, 52))
            z_scaled = z_indep * innovation_scale
            rank_scores = lfilter([1.0], [1.0, -phi], z_scaled, axis=0)

            result_amp = np.empty_like(mapped_amp)
            for k in range(52):
                target_ranks = np.argsort(np.argsort(rank_scores[:, k]))
                sorted_vals = np.sort(mapped_amp[:, k])
                result_amp[:, k] = sorted_vals[target_ranks]

        elif self.mode == AblationMode.QC:
            # Cholesky on i.i.d. scores (no AR(1)), then Iman-Conover
            z_indep = rng.normal(0, 1, (n_samples, 52))
            if self._target_cholesky is not None:
                z_corr = z_indep @ self._target_cholesky.T
            else:
                z_corr = z_indep
            rank_scores = z_corr

            result_amp = np.empty_like(mapped_amp)
            for k in range(52):
                target_ranks = np.argsort(np.argsort(rank_scores[:, k]))
                sorted_vals = np.sort(mapped_amp[:, k])
                result_amp[:, k] = sorted_vals[target_ranks]

        elif self.mode == AblationMode.QTC:
            # Full: AR(1) on Cholesky-correlated scores
            phi = self._phi
            innovation_scale = np.sqrt(max(1.0 - phi ** 2, 0.0))

            z_indep = rng.normal(0, 1, (n_samples, 52))
            if self._target_cholesky is not None:
                z_corr = z_indep @ self._target_cholesky.T
            else:
                z_corr = z_indep
            z_corr *= innovation_scale

            rank_scores = lfilter([1.0], [1.0, -phi], z_corr, axis=0)

            result_amp = np.empty_like(mapped_amp)
            for k in range(52):
                target_ranks = np.argsort(np.argsort(rank_scores[:, k]))
                sorted_vals = np.sort(mapped_amp[:, k])
                result_amp[:, k] = sorted_vals[target_ranks]

        else:
            raise ValueError(f"Unknown ablation mode: {self.mode}")

        # Recombine with phase
        sim_phase = clean_phase.reshape(-1, 52) + self._phase_shift
        sim_complex = result_amp * np.exp(1j * sim_phase)
        sim_complex = sim_complex.reshape(orig_shape)

        return np.stack([sim_complex.real, sim_complex.imag], axis=-1)

    def get_params(self) -> dict:
        return {
            "model": f"M_QTC_ablation_{self.mode.value}",
            "mode": self.mode.value,
            "phi": self._phi,
            "n_quantiles": N_QUANTILES,
            "has_cholesky": self._target_cholesky is not None,
            "calibrated": self._clean_quantiles is not None,
            "seed": self.seed,
        }

    def _apply_quantile_map(self, amp: np.ndarray) -> np.ndarray:
        """Map amplitudes through per-subcarrier quantile transfer."""
        assert self._clean_quantiles is not None
        assert self._jammed_quantiles is not None
        quantile_grid = np.linspace(0, 1, N_QUANTILES)
        mapped = np.empty_like(amp)
        for k in range(52):
            ranks = np.interp(amp[:, k], self._clean_quantiles[k], quantile_grid)
            ranks = np.clip(ranks, 0.0, 1.0)
            mapped[:, k] = np.interp(ranks, quantile_grid, self._jammed_quantiles[k])
        return mapped


def _json_default(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    return x


def split_windows(n_windows, train_fraction=0.7, seed=42):
    """70/30 window-level split via seeded permutation."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_windows)
    n_train = int(n_windows * train_fraction)
    return perm[:n_train], perm[n_train:]


def ablation_single_seed(clean_csi, jammed_csi, seed, train_fraction=0.7):
    """Run all four ablation variants for one train/test split."""
    n_min = min(clean_csi.shape[0], jammed_csi.shape[0])
    clean_csi = clean_csi[:n_min]
    jammed_csi = jammed_csi[:n_min]

    train_idx, test_idx = split_windows(n_min, train_fraction, seed)
    clean_train = clean_csi[train_idx]
    jammed_train = jammed_csi[train_idx]
    clean_test = clean_csi[test_idx]
    jammed_test = jammed_csi[test_idx]

    results = {}
    for mode in ABLATION_VARIANTS:
        model = MQTCAblation(mode=mode, seed=seed)
        model.calibrate(clean_train, jammed_train)
        simulated_test = model.simulate(clean_test)
        scores = aggregate_fidelity_score(simulated_test, jammed_test)

        results[mode.value] = {k: float(scores[k]) for k in METRIC_KEYS}

    return results


def aggregate_across_seeds(seed_results):
    """Mean and std across seed runs for each variant and metric."""
    variant_names = list(seed_results[0].keys())
    aggregated = {}

    for variant in variant_names:
        means = {}
        stds = {}
        for key in METRIC_KEYS:
            values = [sr[variant][key] for sr in seed_results]
            means[key] = float(np.mean(values))
            stds[key] = float(np.std(values))
        aggregated[variant] = {"mean": means, "std": stds}

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

    # Run ablation across all seeds
    seed_results = []
    for i, seed in enumerate(SEEDS):
        t0 = time.time()
        print(f"Seed {seed} ({i + 1}/{len(SEEDS)}) ... ", end="", flush=True)
        sr = ablation_single_seed(clean_csi, jammed_csi, seed=seed)
        elapsed = time.time() - t0
        print(f"done ({elapsed:.1f}s)")

        for variant_name, scores in sr.items():
            agg = scores["aggregate"]
            print(f"  {variant_name:6s}  aggregate={agg:.4f}")
        print()

        seed_results.append(sr)

    # Aggregate mean and std across seeds
    aggregated = aggregate_across_seeds(seed_results)

    # Print summary table
    print("=" * 72)
    print("Table III: M_QTC ablation at 20 dB (controlled room)")
    print("=" * 72)
    header = f"{'Variant':8s}"
    for key in METRIC_KEYS:
        header += f"  {key:>12s}"
    print(header)
    print("-" * 72)

    for variant_name in aggregated:
        row = f"{variant_name:8s}"
        for key in METRIC_KEYS:
            m = aggregated[variant_name]["mean"][key]
            s = aggregated[variant_name]["std"][key]
            row += f"  {m:5.3f}+/-{s:.3f}"
        print(row)
    print("=" * 72)

    # Save results
    output_path = OUTPUT_DIR / "ablation_20dB.json"
    with open(output_path, "w") as f:
        json.dump(aggregated, f, indent=2, default=_json_default)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
