"""Reproduce Table IV: sim-to-real detection AUC at 10 dB in the laboratory.

Trains 5 classifiers on simulated jammed CSI, tests on real jamming.
Per-device protocol across 5 ESP32-C6 receivers, 3 seeds per device.
"""

from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from mqtc.models import M1AWGN, M2PowerScaled, M3Hybrid, MQTC

# --- Configuration ---

DATA_DIR = os.path.join(SCRIPT_DIR, "data", "laboratory", "constant_gaussian_10dB")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "multi_classifier_10dB.json")

DEVICES = [
    "node-acoustic-02",
    "node-infrared-03",
    "node-magnetic-05",
    "node-optical-04",
    "node-thermal-01",
]

SEEDS = [42, 123, 456]

MODELS = [
    ("M_QTC", MQTC),
    ("M3_Hybrid", M3Hybrid),
    ("M2_PowerScaled", M2PowerScaled),
    ("M1_AWGN", M1AWGN),
]

CLASSIFIER_NAMES = ["LogReg", "SVM", "RF", "MLP", "AE"]


class SupervisedAEClassifier:
    """Encoder(52->32->16)-decoder with sigmoid head, trained on MSE+BCE."""

    def __init__(self, seed=42, epochs=100, batch_size=32, lr=1e-3):
        self.seed = seed
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        rng = np.random.default_rng(seed)
        self.params = {
            "W1": rng.normal(0, np.sqrt(2.0 / 52), (52, 32)), "b1": np.zeros(32),
            "W2": rng.normal(0, np.sqrt(2.0 / 32), (32, 16)), "b2": np.zeros(16),
            "W3": rng.normal(0, np.sqrt(2.0 / 16), (16, 32)), "b3": np.zeros(32),
            "W4": rng.normal(0, np.sqrt(2.0 / 32), (32, 52)), "b4": np.zeros(52),
            "Wc": rng.normal(0, np.sqrt(2.0 / 16), (16, 1)), "bc": np.zeros(1),
        }
        self._rng = rng

    @staticmethod
    def _relu(x):
        return np.maximum(x, 0)

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))

    def _forward(self, x):
        p = self.params
        z1 = x @ p["W1"] + p["b1"]; a1 = self._relu(z1)
        z2 = a1 @ p["W2"] + p["b2"]; a2 = self._relu(z2)
        z3 = a2 @ p["W3"] + p["b3"]; a3 = self._relu(z3)
        z4 = a3 @ p["W4"] + p["b4"]
        zc = a2 @ p["Wc"] + p["bc"]
        prob = self._sigmoid(zc)
        return (z1, a1, z2, a2, z3, a3, z4, zc, prob)

    def fit(self, X, y):
        p = self.params
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        n = X.shape[0]
        m = {k: np.zeros_like(v) for k, v in p.items()}
        v = {k: np.zeros_like(val) for k, val in p.items()}
        beta1, beta2, eps, t = 0.9, 0.999, 1e-8, 0

        for _ in range(self.epochs):
            perm = self._rng.permutation(n)
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                xb, yb = X[idx], y[idx]
                bs = xb.shape[0]
                z1, a1, z2, a2, z3, a3, z4, zc, prob = self._forward(xb)

                dz4 = 2.0 * (z4 - xb) / xb.shape[1]
                dW4 = a3.T @ dz4 / bs; db4 = dz4.mean(axis=0)
                da3 = dz4 @ p["W4"].T
                dz3 = da3 * (z3 > 0)
                dW3 = a2.T @ dz3 / bs; db3 = dz3.mean(axis=0)
                da2_recon = dz3 @ p["W3"].T

                dzc = prob - yb
                dWc = a2.T @ dzc / bs; dbc = dzc.mean(axis=0)
                da2_clf = dzc @ p["Wc"].T

                da2 = da2_recon + da2_clf
                dz2 = da2 * (z2 > 0)
                dW2 = a1.T @ dz2 / bs; db2 = dz2.mean(axis=0)
                da1 = dz2 @ p["W2"].T
                dz1 = da1 * (z1 > 0)
                dW1 = xb.T @ dz1 / bs; db1 = dz1.mean(axis=0)

                grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2,
                         "W3": dW3, "b3": db3, "W4": dW4, "b4": db4,
                         "Wc": dWc, "bc": dbc}
                t += 1
                for k, g in grads.items():
                    m[k] = beta1 * m[k] + (1 - beta1) * g
                    v[k] = beta2 * v[k] + (1 - beta2) * g ** 2
                    m_hat = m[k] / (1 - beta1 ** t)
                    v_hat = v[k] / (1 - beta2 ** t)
                    p[k] = p[k] - self.lr * m_hat / (np.sqrt(v_hat) + eps)
        return self

    def predict_proba(self, X):
        prob = self._forward(X)[-1].ravel()
        return np.column_stack([1.0 - prob, prob])


def extract_features(windows):
    """52-dim per-subcarrier mean amplitude per window."""
    csi_complex = windows[..., 0] + 1j * windows[..., 1]
    return np.mean(np.abs(csi_complex), axis=1)


def build_classifiers(seed):
    """Build all 5 binary classifiers."""
    return {
        "LogReg": LogisticRegression(C=1, max_iter=1000),
        "SVM": SVC(kernel="rbf", probability=True, gamma="scale", random_state=seed),
        "RF": RandomForestClassifier(n_estimators=100, random_state=seed),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True,
                             max_iter=500, random_state=seed),
        "AE": SupervisedAEClassifier(seed=seed),
    }


def compute_auc(clf, X_train, y_train, X_test, y_test):
    """Train classifier and return test AUC."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X_train, y_train)
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(X_test)[:, 1]
        else:
            probs = clf.decision_function(X_test)
    try:
        return float(roc_auc_score(y_test, probs))
    except ValueError:
        return float("nan")


def _json_default(obj):
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def main():
    print("=" * 70)
    print("Per-Device Multi-Classifier Sim-to-Real Detection")
    print("Lab environment, constant Gaussian 10 dB")
    print(f"Devices: {len(DEVICES)}, Seeds: {SEEDS}, Classifiers: {CLASSIFIER_NAMES}")
    print("=" * 70)

    all_conditions = ["oracle"] + [name for name, _ in MODELS]
    per_device_aucs = {}

    for device in DEVICES:
        print(f"\n{'='*70}")
        print(f"Device: {device}")
        print(f"{'='*70}")

        baseline_path = os.path.join(DATA_DIR, f"{device}_baseline.npz")
        jammed_path = os.path.join(DATA_DIR, f"{device}_jammed.npz")

        clean_all = np.load(baseline_path)["Xw"]
        jammed_all = np.load(jammed_path)["Xw"]
        print(f"  Clean windows:  {clean_all.shape[0]}")
        print(f"  Jammed windows: {jammed_all.shape[0]}")

        seed_aucs = {
            cond: {clf: [] for clf in CLASSIFIER_NAMES}
            for cond in all_conditions
        }

        for seed in SEEDS:
            print(f"\n  --- Seed {seed} ---")
            rng = np.random.default_rng(seed)

            n_clean = clean_all.shape[0]
            n_jammed = jammed_all.shape[0]

            perm_clean = rng.permutation(n_clean)
            perm_jammed = rng.permutation(n_jammed)

            split_clean = int(n_clean * 0.7)
            split_jammed = int(n_jammed * 0.7)

            clean_train = clean_all[perm_clean[:split_clean]]
            clean_test = clean_all[perm_clean[split_clean:]]
            jammed_train = jammed_all[perm_jammed[:split_jammed]]
            jammed_test = jammed_all[perm_jammed[split_jammed:]]

            print(f"    Clean  train/test: {clean_train.shape[0]} / {clean_test.shape[0]}")
            print(f"    Jammed train/test: {jammed_train.shape[0]} / {jammed_test.shape[0]}")

            feat_clean_train = extract_features(clean_train)
            feat_clean_test = extract_features(clean_test)
            feat_jammed_train = extract_features(jammed_train)
            feat_jammed_test = extract_features(jammed_test)

            scaler = StandardScaler()
            scaler.fit(feat_clean_train)

            feat_clean_train_s = scaler.transform(feat_clean_train)
            feat_clean_test_s = scaler.transform(feat_clean_test)
            feat_jammed_train_s = scaler.transform(feat_jammed_train)
            feat_jammed_test_s = scaler.transform(feat_jammed_test)

            y_test = np.concatenate([
                np.zeros(feat_clean_test_s.shape[0]),
                np.ones(feat_jammed_test_s.shape[0]),
            ])
            X_test = np.vstack([feat_clean_test_s, feat_jammed_test_s])

            # Oracle: train on real jammed
            print("    Oracle (real jammed train)...")
            X_train_oracle = np.vstack([feat_clean_train_s, feat_jammed_train_s])
            y_train_oracle = np.concatenate([
                np.zeros(feat_clean_train_s.shape[0]),
                np.ones(feat_jammed_train_s.shape[0]),
            ])

            oracle_clfs = build_classifiers(seed)
            for clf_name, clf in oracle_clfs.items():
                auc = compute_auc(clf, X_train_oracle, y_train_oracle, X_test, y_test)
                seed_aucs["oracle"][clf_name].append(auc)
                print(f"      Oracle {clf_name}: AUC = {auc:.3f}")

            # Simulation models
            for model_name, model_cls in MODELS:
                print(f"    {model_name}...")
                model = model_cls(seed=seed)
                model.calibrate(clean_train, jammed_train)

                sim_jammed_train = model.simulate(clean_train)
                feat_sim_train = extract_features(sim_jammed_train)
                feat_sim_train_s = scaler.transform(feat_sim_train)

                X_train_sim = np.vstack([feat_clean_train_s, feat_sim_train_s])
                y_train_sim = np.concatenate([
                    np.zeros(feat_clean_train_s.shape[0]),
                    np.ones(feat_sim_train_s.shape[0]),
                ])

                clfs = build_classifiers(seed)
                for clf_name, clf in clfs.items():
                    auc = compute_auc(clf, X_train_sim, y_train_sim, X_test, y_test)
                    seed_aucs[model_name][clf_name].append(auc)
                    print(f"      {clf_name}: AUC = {auc:.3f}")

        per_device_aucs[device] = seed_aucs

    # Aggregate results
    print("\n" + "=" * 70)
    print("Aggregating results...")

    results = {"per_device": {}, "mean_across_devices": {}}

    for device in DEVICES:
        results["per_device"][device] = {}
        for cond in all_conditions:
            results["per_device"][device][cond] = {}
            for clf_name in CLASSIFIER_NAMES:
                aucs = per_device_aucs[device][cond][clf_name]
                results["per_device"][device][cond][clf_name] = {
                    "auc_mean": float(np.nanmean(aucs)),
                    "auc_std": float(np.nanstd(aucs)),
                    "auc_per_seed": [float(a) for a in aucs],
                }

    for cond in all_conditions:
        results["mean_across_devices"][cond] = {}
        for clf_name in CLASSIFIER_NAMES:
            device_means = [
                results["per_device"][d][cond][clf_name]["auc_mean"]
                for d in DEVICES
            ]
            results["mean_across_devices"][cond][clf_name] = {
                "auc_mean": float(np.nanmean(device_means)),
                "auc_std": float(np.nanstd(device_means)),
            }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=_json_default)
    print(f"\nResults saved to {OUTPUT_PATH}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Mean AUC across 5 devices (per-device mean +/- std across devices)")
    print("=" * 70)

    header = f"{'Classifier':<10}"
    for cond in all_conditions:
        header += f" | {cond:>12}"
    print(header)
    print("-" * len(header))

    for clf_name in CLASSIFIER_NAMES:
        row = f"{clf_name:<10}"
        for cond in all_conditions:
            m = results["mean_across_devices"][cond][clf_name]["auc_mean"]
            s = results["mean_across_devices"][cond][clf_name]["auc_std"]
            row += f" | {m:>5.3f}+/-{s:.3f}"
        print(row)

    print("-" * len(header))
    row = f"{'Mean':<10}"
    for cond in all_conditions:
        means = [
            results["mean_across_devices"][cond][c]["auc_mean"]
            for c in CLASSIFIER_NAMES
        ]
        row += f" | {np.mean(means):>11.3f} "
    print(row)

    # Per-device detail
    print("\n" + "=" * 70)
    print("PER-DEVICE DETAIL (mean across 3 seeds)")
    print("=" * 70)

    for device in DEVICES:
        print(f"\n  {device}:")
        header_d = f"  {'Classifier':<10}"
        for cond in all_conditions:
            header_d += f" | {cond:>12}"
        print(header_d)
        print("  " + "-" * (len(header_d) - 2))
        for clf_name in CLASSIFIER_NAMES:
            row = f"  {clf_name:<10}"
            for cond in all_conditions:
                m = results["per_device"][device][cond][clf_name]["auc_mean"]
                row += f" | {m:>12.3f}"
            print(row)

    print("\n" + "=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
