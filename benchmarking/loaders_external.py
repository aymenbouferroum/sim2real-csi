"""Loaders that turn external CSI datasets into [N, frames, K, 2] real/imag format."""

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

# Wallhack1.8k: 52 L-LTF subcarrier indices (from authors' datasets.py)
_WALLHACK_SUBCARRIERS = [i for i in range(6, 32)] + [i for i in range(33, 59)]


def load_wallhack_csv(path: str) -> np.ndarray:
    """Parse one Wallhack CSV into complex CSI [n_packets, 52]."""
    df = pd.read_csv(path)
    raw = df["data"].to_numpy()

    rows = []
    for s in raw:
        vals = s[1:-1].split(",")
        rows.append(np.array(vals, dtype=np.int32))
    raw_len = min(r.shape[0] for r in rows)
    arr = np.stack([r[:raw_len] for r in rows], axis=0)

    cols_real = [idx * 2 for idx in _WALLHACK_SUBCARRIERS]
    cols_imag = [idx * 2 - 1 for idx in _WALLHACK_SUBCARRIERS]
    csi = arr[:, cols_real].astype(np.float64) + 1j * arr[:, cols_imag].astype(np.float64)
    return csi


def _window_complex(csi: np.ndarray, frames: int = 32) -> np.ndarray:
    """Window [n_packets, K] complex CSI into [N, frames, K, 2]."""
    n, k = csi.shape
    n_win = n // frames
    csi = csi[: n_win * frames].reshape(n_win, frames, k)
    return np.stack([csi.real, csi.imag], axis=-1)


def load_wallhack_pair(
    root: str,
    domain: str = "NLOS/BQ",
    clean_prefix: str = "b",
    modified_prefixes=("w",),
    frames: int = 32,
    seed: int = 0,
):
    """Build paired (clean, modified) arrays for one Wallhack domain cell."""
    cell = Path(root) / domain
    files = sorted(cell.glob("*.csv"))

    def _match(fname: str, prefix: str) -> bool:
        stem = fname[:-4]
        return stem.startswith(prefix) and stem[len(prefix):].isdigit()

    clean_files = [f for f in files if _match(f.name, clean_prefix)]
    mod_files = [f for f in files
                 if any(_match(f.name, p) for p in modified_prefixes)]

    clean_csi = np.concatenate([load_wallhack_csv(str(f)) for f in clean_files], axis=0)
    mod_csi = np.concatenate([load_wallhack_csv(str(f)) for f in mod_files], axis=0)

    clean_w = _window_complex(clean_csi, frames)
    mod_w = _window_complex(mod_csi, frames)

    # shuffle so truncation to min-length samples representatively
    rng = np.random.default_rng(seed)
    clean_w = clean_w[rng.permutation(clean_w.shape[0])]
    mod_w = mod_w[rng.permutation(mod_w.shape[0])]

    return clean_w, mod_w


def _label_windows(files_labels, loader, frames: int):
    """Window each (file, label) pair; returns (X [N, frames, K, 2], y [N])."""
    Xs, ys = [], []
    for path, label in files_labels:
        csi = loader(str(path))
        w = _window_complex(csi, frames)
        if w.shape[0] == 0:
            continue
        Xs.append(w)
        ys.append(np.full(w.shape[0], label, dtype=int))
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)


_WALLHACK_CLASS = {"b": 0, "w": 1, "ww": 2}


def load_wallhack_labeled(root: str, cell: str = "LOS/BQ", frames: int = 32):
    """Wallhack domain cell -> (X, y) for 3-class activity recognition."""
    cell_dir = Path(root) / cell
    files_labels = []
    for f in sorted(cell_dir.glob("*.csv")):
        stem = f.stem
        for pref in ("ww", "w", "b"):
            if stem.startswith(pref) and stem[len(pref):].isdigit():
                files_labels.append((f, _WALLHACK_CLASS[pref]))
                break
    return _label_windows(files_labels, load_wallhack_csv, frames)


def _signfi_to_windows(csi4d: np.ndarray, antenna: int, frames: int) -> np.ndarray:
    """SignFi [200, 30, 3, N] complex -> windows [Nw, frames, 30, 2]."""
    arr = np.transpose(csi4d[:, :, antenna, :], (2, 0, 1))
    n_inst, t, k = arr.shape
    n_win_per = t // frames
    arr = arr[:, : n_win_per * frames, :].reshape(n_inst, n_win_per, frames, k)
    arr = arr.reshape(n_inst * n_win_per, frames, k)
    return np.stack([arr.real, arr.imag], axis=-1)


def load_signfi_pair(
    lab_path: str,
    home_path: str,
    antenna: int = 0,
    frames: int = 32,
    max_windows: int | None = 3000,
    seed: int = 0,
):
    """Build paired (clean=lab, modified=home) SignFi arrays."""
    lab = sio.loadmat(lab_path)
    home = sio.loadmat(home_path)
    csid_lab = lab["csid_lab"]
    csid_home = home["csid_home"]

    if not (np.iscomplexobj(csid_lab) and np.iscomplexobj(csid_home)):
        raise ValueError("SignFi CSI expected complex; got real arrays")

    clean = _signfi_to_windows(csid_lab, antenna, frames)
    modified = _signfi_to_windows(csid_home, antenna, frames)

    if max_windows is not None:
        rng = np.random.default_rng(seed)
        if clean.shape[0] > max_windows:
            clean = clean[rng.permutation(clean.shape[0])[:max_windows]]
        if modified.shape[0] > max_windows:
            modified = modified[rng.permutation(modified.shape[0])[:max_windows]]

    return clean, modified


def _widar_load_stream(files, antenna: int) -> np.ndarray:
    """Parse Intel-5300 .dat files -> complex [total_pkts, 30]."""
    import csiread
    chunks = []
    for f in files:
        dev = csiread.Intel(str(f), nrxnum=3, ntxnum=1)
        dev.read()
        csi = dev.get_scaled_csi()
        chunks.append(csi[:, :, antenna, 0])
    return np.concatenate(chunks, axis=0)


def _shuffle_subsample(clean, modified, max_windows, seed):
    rng = np.random.default_rng(seed)
    clean = clean[rng.permutation(clean.shape[0])]
    modified = modified[rng.permutation(modified.shape[0])]
    if max_windows is not None:
        clean, modified = clean[:max_windows], modified[:max_windows]
    return clean, modified


def load_signfi_labeled(mat_path: str, csi_var: str, label_var: str,
                        antenna: int = 0, frames: int = 32, max_per_class=None,
                        seed: int = 0):
    """SignFi domain (lab or home) -> (X, y) for 276-class sign recognition."""
    m = sio.loadmat(mat_path)
    csi = m[csi_var]
    labels = m[label_var].ravel().astype(int)
    arr = np.transpose(csi[:, :, antenna, :], (2, 0, 1))
    Xs, ys = [], []
    for i in range(arr.shape[0]):
        w = _window_complex(arr[i], frames)
        if w.shape[0]:
            Xs.append(w)
            ys.append(np.full(w.shape[0], labels[i], dtype=int))
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    if max_per_class is not None:
        X, y = _subsample_per_class(X, y, max_per_class, seed)
    return X, y


def load_widar_labeled(root: str, date_folder: str, receiver: str = "r1",
                       antenna: int = 0, frames: int = 32, max_per_class=None,
                       seed: int = 0):
    """Widar date-folder (one room) -> (X, y) for 6-class gesture recognition."""
    user_dir = next((Path(root) / date_folder).glob("user*"))
    files_labels = []
    for f in sorted(user_dir.glob(f"*-{receiver}.dat")):
        parts = f.stem.split("-")
        if len(parts) == 6:
            files_labels.append((f, int(parts[1])))

    def _one(path):
        import csiread
        dev = csiread.Intel(str(path), nrxnum=3, ntxnum=1)
        dev.read()
        return dev.get_scaled_csi()[:, :, antenna, 0]

    X, y = _label_windows(files_labels, _one, frames)
    if max_per_class is not None:
        X, y = _subsample_per_class(X, y, max_per_class, seed)
    return X, y


def _subsample_per_class(X, y, max_per_class, seed):
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        if len(idx) > max_per_class:
            idx = rng.permutation(idx)[:max_per_class]
        keep.append(idx)
    keep = np.concatenate(keep)
    keep = rng.permutation(keep)
    return X[keep], y[keep]


def load_widar_pair(
    root: str,
    clean_orient: int = 1,
    modified_orient: int = 3,
    receiver: str = "r1",
    antenna: int = 0,
    frames: int = 32,
    max_windows: int | None = 3000,
    seed: int = 0,
):
    """Build paired (clean, modified) Widar3.0 arrays by face orientation."""
    user_dir = next(Path(root).glob("2018112*/user*"))

    def _files_for(orient: int):
        out = []
        for f in sorted(user_dir.glob(f"*-{receiver}.dat")):
            parts = f.stem.split("-")
            if len(parts) == 6 and parts[3] == str(orient):
                out.append(f)
        return out

    clean_csi = _widar_load_stream(_files_for(clean_orient), antenna)
    mod_csi = _widar_load_stream(_files_for(modified_orient), antenna)

    clean = _window_complex(clean_csi, frames)
    modified = _window_complex(mod_csi, frames)
    return _shuffle_subsample(clean, modified, max_windows, seed)


def load_widar_crossroom(
    root: str,
    room1: str,
    room2: str,
    receiver: str = "r1",
    antenna: int = 0,
    frames: int = 32,
    max_windows: int | None = 3000,
    seed: int = 0,
):
    """Widar3.0 cross-room environment domain shift."""
    def _room_files(date_folder: str):
        d = next((Path(root) / date_folder).glob("user*"))
        return sorted(d.glob(f"*-{receiver}.dat"))

    clean_csi = _widar_load_stream(_room_files(room1), antenna)
    mod_csi = _widar_load_stream(_room_files(room2), antenna)

    clean = _window_complex(clean_csi, frames)
    modified = _window_complex(mod_csi, frames)
    return _shuffle_subsample(clean, modified, max_windows, seed)
