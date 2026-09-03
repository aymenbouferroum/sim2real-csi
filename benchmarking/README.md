# Benchmarking

Reproduces the external benchmark table: M_QTC vs 4 published CSI augmentation methods on 3 external datasets.

## What this reproduces

| Method | Wallhack1.8k | SignFi | Widar 3.0 |
|--------|---:|---:|---:|
| **M_QTC** | **0.19** | **0.96** | **0.06** |
| Gao et al. (AWGN) | 0.55 | 14.41 | 1.31 |
| AirFi (Wang et al.) | 0.37 | 13.83 | 1.30 |
| Strohmayer et al. | 0.51 | 8.28 | 0.34 |
| Serbetci et al. | 0.55 | 10.23 | 0.60 |

Metric: amplitude Wasserstein distance (lower = more realistic). 5 seeds, 70/30 train/test split.

## Datasets

| Dataset | Hardware | Subcarriers | Clean | Modified | Size | Source |
|---------|----------|:-----------:|-------|----------|------|--------|
| Wallhack1.8k | ESP32 | 52 | empty room | person walking/waving | 135 MB | [Zenodo 15147388](https://zenodo.org/records/15147388) |
| SignFi | Intel 5300 | 30 | lab recordings | home recordings (same signs) | 2.8 GB | [github yongsen/SignFi](https://github.com/yongsen/SignFi) |
| Widar 3.0 | Intel 5300 | 30 | Room#1 (classroom) | Room#2 (hall) | 4.5 GB | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/2760bb9557ca4d09a74d/) |

## Quick start

### Full run (downloads data + runs benchmark + prints table)

```bash
cd benchmarking
pip install numpy scipy statsmodels scikit-learn pandas csiread
python reproduce_table.py
```

This takes ~15 minutes (mostly download time). Output: result JSONs in `results/` + the printed table.

### If data is already downloaded

```bash
python reproduce_table.py --skip-download
```

### Just print the table from saved results

```bash
python reproduce_table.py --table-only
```

## File structure

```
benchmarking/
├── README.md                     # this file
├── reproduce_table.py            # one-command reproducibility script
├── mqtc/                         # core modules (metrics + models)
│   ├── models/
│   │   ├── base.py               # SimulationModel ABC
│   │   ├── m1_awgn.py            # AWGN baseline (Gao et al.)
│   │   └── m_qtc.py              # M_QTC (52 subcarriers)
│   ├── metrics/
│   │   ├── aggregate.py          # 4-metric fidelity scorer
│   │   ├── amplitude.py          # Wasserstein distance per subcarrier
│   │   ├── phase.py              # circular variance
│   │   ├── temporal.py           # ACF difference
│   │   └── spectral.py           # correlation matrix Frobenius
│   └── data/
│       └── loaders.py            # NPZ CSI loader
├── mqtc_flexible.py              # M_QTC generalized to any subcarrier count
├── competitor_methods.py         # AirFi, ProFiNet, Strohmayer, Serbetci
├── external_models_flexible.py   # calibrated Strohmayer/Serbetci (flexible)
├── external_benchmark.py         # generic benchmark runner
├── loaders_external.py           # Wallhack/SignFi/Widar data parsers
├── results/                      # saved result JSONs
│   ├── wallhack_NLOS_BQ.json
│   ├── signfi_antenna1.json
│   └── widar_crossroom.json
└── data/                         # external datasets (created by reproduce_table.py)
    ├── wallhack/
    ├── signfi/
    └── widar/
```

## Methods

Each method is a `calibrate(clean, modified) -> simulate(clean)` transform:

- **M_QTC**: per-subcarrier quantile mapping + AR(1) temporal filtering + Iman-Conover copula reordering. Learns the actual distributional transformation.
- **Gao et al. (AWGN)**: additive complex Gaussian noise at power-matched SNR.
- **AirFi (Wang et al.)**: additive Gaussian on amplitude only, per-subcarrier variance-matched.
- **Strohmayer et al.**: per-subcarrier calibrated amplitude scaling (ratio of modified/clean mean).
- **Serbetci et al.**: calibrated global phase rotation + amplitude dB shift.

All competitors are given their most generous calibrated form (fitted to the real modified data).

## Protocol

For each dataset, for each method, repeated over 5 random seeds:
1. Split paired data 70/30 (same split for all methods).
2. Calibrate on train pairs.
3. Simulate: transform clean_test into synthetic modified.
4. Score: amplitude Wasserstein distance between synthetic and real modified_test.

Calibration uses only the train split and scoring uses only the held-out test split. The split is deterministic per seed, so every method is evaluated on the identical train/test partition (see `split_windows` and `run_paired_benchmark` in `external_benchmark.py`).
