# CSI Dataset

Dataset for "CSI Simulation: Why Additive Noise Fails and How to Fix It".

WiFi Channel State Information (CSI) collected from ESP32-C6 receivers under baseline (no jamming) and jammed (constant Gaussian interference) conditions. Two environments: a controlled single-receiver room and a multi-receiver laboratory.


## Dataset Format

All files are NumPy `.npz` archives containing a single key `Xw`.

- **Array shape**: `[N_windows, 32, 52, 2]`
  - `N_windows`: number of observation windows (varies per file)
  - `32`: frames per window, collected at approximately 3 frames/second (roughly 10.7 seconds per window)
  - `52`: OFDM subcarriers (HT20 indices 4 through 55, guard bands and DC subcarrier removed)
  - `2`: real and imaginary parts of the complex CSI coefficient
- **Data type**: float32
- **Complex reconstruction**: `csi = data[..., 0] + 1j * data[..., 1]`

Example loading code:

```python
import numpy as np

data = np.load("controlled/constant_gaussian_20dB/baseline.npz")
Xw = data["Xw"]                          # shape: (409, 32, 52, 2)
csi = Xw[..., 0] + 1j * Xw[..., 1]      # complex CSI, shape: (409, 32, 52)
amplitude = np.abs(csi)                   # CSI amplitude
phase = np.angle(csi)                     # CSI phase
```


## Hardware

| Component | Details |
|-----------|---------|
| Receiver | ESP32-C6 running ESP32-CSI-Tool, CSI streamed via MQTT |
| Access point | Raspberry Pi 5, 802.11n HT20, channel 6 (2.437 GHz) |
| Jammer | HackRF One, constant Gaussian waveform via `hackrf_transfer` |
| Waveform generation | JamRF |


## Environments

### Controlled room (`controlled/`)

Single ESP32-C6 receiver at a fixed line-of-sight position with no ambient WiFi. The jammer operated at 20 dB IF gain with a constant Gaussian waveform.

| File | Windows | Description |
|------|---------|-------------|
| `baseline.npz` | 409 | No jamming active |
| `jammed.npz` | 307 | 20 dB constant Gaussian jamming |

### Laboratory (`laboratory/`)

Five ESP32-C6 receivers at different positions and orientations in an active research lab. The environment includes ambient WiFi, human movement, and multipath propagation. Data was collected using interleaved 3-minute clean/jammed blocks at three IF gain levels: 10, 15, and 20 dB.

Each power level directory contains per-device files and a merged file that concatenates all five devices.

**10 dB** (`laboratory/constant_gaussian_10dB/`):

| File | Windows |
|------|---------|
| `node-thermal-01_baseline.npz` | 165 |
| `node-thermal-01_jammed.npz` | 113 |
| `node-acoustic-02_baseline.npz` | 157 |
| `node-acoustic-02_jammed.npz` | 102 |
| `node-infrared-03_baseline.npz` | 158 |
| `node-infrared-03_jammed.npz` | 109 |
| `node-optical-04_baseline.npz` | 179 |
| `node-optical-04_jammed.npz` | 119 |
| `node-magnetic-05_baseline.npz` | 160 |
| `node-magnetic-05_jammed.npz` | 107 |
| `merged_baseline.npz` | 819 |
| `merged_jammed.npz` | 550 |

**15 dB** (`laboratory/constant_gaussian_15dB/`):

| File | Windows |
|------|---------|
| `node-thermal-01_baseline.npz` | 158 |
| `node-thermal-01_jammed.npz` | 107 |
| `node-acoustic-02_baseline.npz` | 158 |
| `node-acoustic-02_jammed.npz` | 105 |
| `node-infrared-03_baseline.npz` | 154 |
| `node-infrared-03_jammed.npz` | 103 |
| `node-optical-04_baseline.npz` | 180 |
| `node-optical-04_jammed.npz` | 119 |
| `node-magnetic-05_baseline.npz` | 148 |
| `node-magnetic-05_jammed.npz` | 97 |
| `merged_baseline.npz` | 798 |
| `merged_jammed.npz` | 531 |

**20 dB** (`laboratory/constant_gaussian_20dB/`):

| File | Windows |
|------|---------|
| `node-thermal-01_baseline.npz` | 155 |
| `node-thermal-01_jammed.npz` | 101 |
| `node-acoustic-02_baseline.npz` | 158 |
| `node-acoustic-02_jammed.npz` | 104 |
| `node-infrared-03_baseline.npz` | 167 |
| `node-infrared-03_jammed.npz` | 114 |
| `node-optical-04_baseline.npz` | 180 |
| `node-optical-04_jammed.npz` | 119 |
| `node-magnetic-05_baseline.npz` | 159 |
| `node-magnetic-05_jammed.npz` | 108 |
| `merged_baseline.npz` | 819 |
| `merged_jammed.npz` | 546 |


## Device Naming

| Code name | Role |
|-----------|------|
| node-thermal-01 | Controlled room receiver; lab receiver 1 |
| node-acoustic-02 | Lab receiver 2 |
| node-infrared-03 | Lab receiver 3 |
| node-optical-04 | Lab receiver 4 |
| node-magnetic-05 | Lab receiver 5 |

All devices are ESP32-C6 boards running identical firmware.


## Directory Structure

```
data/
  controlled/
    constant_gaussian_20dB/
      baseline.npz
      jammed.npz
  laboratory/
    constant_gaussian_10dB/
      merged_baseline.npz
      merged_jammed.npz
      node-acoustic-02_baseline.npz
      node-acoustic-02_jammed.npz
      node-infrared-03_baseline.npz
      node-infrared-03_jammed.npz
      node-magnetic-05_baseline.npz
      node-magnetic-05_jammed.npz
      node-optical-04_baseline.npz
      node-optical-04_jammed.npz
      node-thermal-01_baseline.npz
      node-thermal-01_jammed.npz
    constant_gaussian_15dB/
      merged_baseline.npz
      merged_jammed.npz
      node-acoustic-02_baseline.npz
      node-acoustic-02_jammed.npz
      node-infrared-03_baseline.npz
      node-infrared-03_jammed.npz
      node-magnetic-05_baseline.npz
      node-magnetic-05_jammed.npz
      node-optical-04_baseline.npz
      node-optical-04_jammed.npz
      node-thermal-01_baseline.npz
      node-thermal-01_jammed.npz
    constant_gaussian_20dB/
      merged_baseline.npz
      merged_jammed.npz
      node-acoustic-02_baseline.npz
      node-acoustic-02_jammed.npz
      node-infrared-03_baseline.npz
      node-infrared-03_jammed.npz
      node-magnetic-05_baseline.npz
      node-magnetic-05_jammed.npz
      node-optical-04_baseline.npz
      node-optical-04_jammed.npz
      node-thermal-01_baseline.npz
      node-thermal-01_jammed.npz
```


## Ethics

No personal data was collected. All interference experiments were conducted in controlled indoor environments with authorization. No ambient users were affected.
