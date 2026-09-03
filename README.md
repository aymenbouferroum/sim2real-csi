# CSI Simulation: Why Additive Noise Fails and How to Fix It

**Aymen Bouferroum, Ildi Alla, Vincent Lenders, Valeria Loscri**

## Abstract

Channel State Information (CSI) has become a widely used wireless channel sensing modality. Training CSI-based sensing models often relies on simulated data produced by adding white Gaussian noise (AWGN) to recorded channel estimates. This practice assumes that the receiver chain between the antenna and the channel estimator is linear and gain-invariant. We test this assumption empirically using RF jamming as a controlled perturbation on 6 commodity receivers across 2 indoor environments. The assumption does not hold. Automatic gain control compresses the channel estimate multiplicatively before digitization, producing amplitude distributions that no additive noise variance can reproduce. To close the resulting fidelity gap, we propose M_QTC, a measurement-calibrated model that learns the per-subcarrier distribution transformation through quantile mapping, temporal filtering, and copula-based cross-subcarrier reordering. M_QTC reduces amplitude error 8-fold and closes 89% of the aggregate fidelity gap across four complementary dimensions. The improvement transfers directly to downstream tasks, where 5 classifiers from different families trained on M_QTC-simulated data achieve 93% of jamming detection performance, while AWGN-trained classifiers remain near random decision.

## About This Artifact

This repository contains the complete source code, processed CSI dataset, and evaluation scripts for M_QTC. The dataset is included directly (~11 MB). A single shell command (`./reproduce.sh`) reproduces all data-driven paper figures and experiment-result tables from the shipped data. Pre-computed results are included so that tables and figures can be regenerated without re-running experiments.

## Headline Results

| Metric | Value |
|:-------|------:|
| Amplitude fidelity improvement over AWGN | 8x (Wasserstein 3.09 to 0.39) |
| Aggregate fidelity gap closure (4 dimensions) | 89% |
| Dominant mechanism (ablation) | Copula reordering (9x) |
| Per-device consistency (5 lab receivers) | Aggregate 1.11--1.27 |
| Sim-to-real detection at 10 dB, M_QTC mean AUC | 0.904 (93% of oracle 0.967) |
| Sim-to-real detection at 10 dB, AWGN mean AUC | 0.522 (near chance) |
| External validation (3 public datasets, 2 platforms) | Lowest amplitude error on all 3 |

## Requirements

- **Python:** >= 3.10
- **Disk:** ~12 MB (dataset included)
- **GPU:** not required
- **OS:** Linux, macOS, or Windows with Python

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# 1. Download and extract the repository, then enter the directory
cd sim2real-csi

# 2. Install dependencies
pip install -r requirements.txt

# 3. Reproduce all paper results (~15 min)
./reproduce.sh
```

## Reproduction

The `reproduce.sh` script is the single entry point for artifact evaluation. It runs the experiment scripts, generates result tables, and renders the data-driven paper figures.

```bash
./reproduce.sh
```

Result tables are printed to the terminal. Figures are saved as PNG in `figures_out/`.

## Paper-to-Artifact Mapping

| # | Paper Section | Output | Command |
|:-:|:--------------|:-------|:--------|
| 1 | Sec. VI-A | Table II (within-scenario fidelity) | `python3 tables/print_table_II.py` |
| 2 | Sec. VI-A | Figure 4 (amplitude profile, 52 subcarriers) | `cd "Figures/Figure 4 - Amplitude Profile" && python3 fig_amplitude_profile_trial.py` |
| 3 | Sec. VI-A | Figure 5 (amplitude KDE, subcarrier 26) | `cd "Figures/Figure 5 - Amplitude KDE" && python3 fig_amplitude_kde.py` |
| 4 | Sec. VI-B | Table III (M_QTC ablation) | `python3 tables/print_table_III.py` |
| 5 | Sec. VI-C | Figure 6 (per-device fidelity) | `cd "Figures/Figure 6 - Per Device Fidelity" && python3 fig_perdevice_fidelity.py` |
| 6 | Sec. VI-D | Table IV (sim-to-real detection AUC) | `python3 tables/print_table_IV.py` |
| 7 | Sec. VI-E | Table V (external validation, 3 datasets) | `python3 tables/print_table_V.py` |

### Experiment Scripts

| Script | Produces | Runtime |
|:-------|:---------|:--------|
| `python3 run_within_scenario.py` | `results/controlled_20dB.json` (Table II) | ~2 min |
| `python3 run_ablation.py` | `results/ablation_20dB.json` (Table III) | ~2 min |
| `python3 run_multi_classifier.py` | `results/multi_classifier_10dB.json` (Table IV) | ~10 min |
| `python3 run_per_device_fidelity.py` | `results/per_device_20dB/*.json` (Figure 6) | ~1 min |

## Directory Layout

```
sim2real-csi/
|-- README.md
|-- requirements.txt
|-- reproduce.sh                   Single-command reproduction driver
|
|-- run_within_scenario.py         Experiment: within-scenario fidelity (Table II)
|-- run_ablation.py                Experiment: component ablation (Table III)
|-- run_multi_classifier.py        Experiment: sim-to-real detection (Table IV)
|-- run_per_device_fidelity.py     Experiment: per-device consistency (Figure 6)
|
|-- mqtc/                          Core Python package
|   |-- models/
|   |   |-- base.py                SimulationModel ABC
|   |   |-- m1_awgn.py             M1: additive white Gaussian noise
|   |   |-- m2_power_scaled.py     M2: per-subcarrier scaling
|   |   |-- m3_hybrid.py           M3: scaling + correlation
|   |   `-- m_qtc.py               M_QTC: quantile-temporal-copula
|   |-- metrics/
|   |   |-- amplitude.py           Wasserstein distance per subcarrier
|   |   |-- phase.py               Circular variance per subcarrier
|   |   |-- temporal.py            ACF difference per subcarrier
|   |   |-- spectral.py            Correlation matrix Frobenius distance
|   |   `-- aggregate.py           Composite fidelity score (4 metrics)
|   `-- data/
|       `-- loaders.py             NPZ data loader
|
|-- data/                          CSI dataset (included, ~11 MB)
|   |-- README.md                  Detailed dataset documentation
|   |-- controlled/                Single-receiver controlled room (20 dB)
|   `-- laboratory/                Five-receiver lab (10, 15, 20 dB)
|
|-- results/                       Pre-computed experiment results (JSON)
|   |-- controlled_20dB.json       Table II
|   |-- ablation_20dB.json         Table III (controlled room)
|   |-- ablation_lab_20dB.json     Table III (laboratory)
|   |-- multi_classifier_10dB.json Table IV
|   `-- per_device_20dB/           Figure 6 (one JSON per receiver)
|
|-- tables/                        Experiment-result table printers
|-- Figures/                       One directory per data-driven figure
|-- benchmarking/                  External benchmark module (Table V)
|   |-- reproduce_table.py         One-command benchmark driver
|   `-- results/                   Pre-computed benchmark results
`-- figures_out/                   Generated figures
```

## Dataset

All CSI data is included in `data/` (~11 MB). See `data/README.md` for the complete file inventory and per-scenario window counts.

**Format.** NumPy `.npz` archives containing a single key `Xw` with shape `[N_windows, 32, 52, 2]`. The last dimension holds real and imaginary parts of complex CSI. Reconstruction: `csi = data[..., 0] + 1j * data[..., 1]`. Each window contains 32 consecutive frames at approximately 3 frames per second (~10.7 seconds per window), with 52 OFDM subcarriers (HT20, guard bands and DC removed).

**Hardware.** ESP32-C6 receivers running ESP32-CSI-Tool (CSI streamed via MQTT), Raspberry Pi 5 access point (802.11n HT20, channel 6), HackRF One jammer with constant Gaussian waveforms generated by JamRF.

**Two environments:**

- **Controlled room** (`data/controlled/`): Single receiver, fixed line-of-sight position, no ambient WiFi. One power level (20 dB IF gain). Used for fidelity evaluation (Tables II, III) and Figures 4, 5.

- **Laboratory** (`data/laboratory/`): Five ESP32-C6 receivers at different positions in an active research lab with ambient WiFi, human movement, and multipath. Three power levels (10, 15, 20 dB IF gain) collected in interleaved 3-minute clean/jammed blocks. Used for per-device consistency (Figure 6), sim-to-real detection (Table IV), and ablation lab column (Table III).

## External Benchmarking (Table V)

The `benchmarking/` directory compares M_QTC against four published CSI augmentation methods on three external public datasets spanning two hardware platforms (ESP32 and Intel 5300). Download each dataset and place it under `benchmarking/data/`:

| Dataset | Hardware | Subcarriers | Place in | Source |
|:--------|:---------|:-----------:|:---------|:-------|
| Wallhack1.8k | ESP32 | 52 | `benchmarking/data/wallhack/` | [Zenodo 15147388](https://zenodo.org/records/15147388) |
| SignFi | Intel 5300 | 30 | `benchmarking/data/signfi/` | [github yongsen/SignFi](https://github.com/yongsen/SignFi) |
| Widar 3.0 | Intel 5300 | 30 | `benchmarking/data/widar/` | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/2760bb9557ca4d09a74d/) |

Then run:

```bash
cd benchmarking
python3 reproduce_table.py
```

Pre-computed results are shipped in `benchmarking/results/`. See `benchmarking/README.md` for details.

## Expected Output

Running `./reproduce.sh` prints four LaTeX table bodies (Tables II through V) to the terminal and writes three PNG figures to `figures_out/`.

## Citation

If you use this code or dataset, please cite:

```bibtex
@inproceedings{bouferroum2026csisimulation,
  title     = {{CSI} Simulation: Why Additive Noise Fails and How to Fix It},
  author    = {Bouferroum, Aymen and Alla, Ildi and Lenders, Vincent and Loscri, Valeria},
  booktitle = {Proc. 28th IEEE Int. Conf. Modeling, Analysis and Simulation of Wireless and Mobile Systems (MSWiM)},
  year      = {2026},
  note      = {arXiv:2607.01882}
}
```

**Paper:** [arXiv:2607.01882](https://arxiv.org/abs/2607.01882) (to appear in MSWiM 2026)

## License

License: [GNU Affero General Public License v3.0 (AGPL-3.0)](https://www.gnu.org/licenses/agpl-3.0.en.html). See the [LICENSE](LICENSE) file for the full text.
