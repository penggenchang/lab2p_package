from __future__ import annotations
from pathlib import Path
import numpy as np

def load_good_frames(plane0: Path) -> np.ndarray | None:
    p = Path(plane0) / "bad_frames_final.npy"
    if not p.exists():
        return None
    return ~np.load(p).astype(bool).ravel()

def load_suite2p(plane0: Path):
    plane0 = Path(plane0)
    ops = np.load(plane0 / "ops.npy", allow_pickle=True).item()
    F = np.load(plane0 / "F.npy", mmap_mode="r")
    Fneu = np.load(plane0 / "Fneu.npy", mmap_mode="r")
    iscell = np.load(plane0 / "iscell.npy", allow_pickle=True)
    iscell = iscell[:, 0].astype(bool) if iscell.ndim == 2 else iscell.astype(bool)
    return F, Fneu, ops, iscell

def compute_F0_percentile(Fcorr: np.ndarray, ops: dict, win_s=60, pctl=20):
    fs = float(ops.get("fs", 10.0))
    win = max(3, (int(win_s * fs) | 1))  # odd window
    try:
        from scipy.ndimage import percentile_filter
        return percentile_filter(Fcorr, size=(1, win), percentile=pctl, mode="nearest")
    except Exception:
        alpha = 1.0 - np.exp(-1.0 / win)
        F0 = np.zeros_like(Fcorr, dtype=np.float32)
        F0[:, 0] = Fcorr[:, 0]
        for t in range(1, Fcorr.shape[1]):
            F0[:, t] = (1 - alpha) * F0[:, t - 1] + alpha * np.minimum(Fcorr[:, t], F0[:, t - 1])
        return F0

def compute_dff(Fcorr: np.ndarray, F0: np.ndarray, eps=1e-5, F0_floor=1.0, clip_min=None):
    F0c = np.maximum(F0, F0_floor)
    dff = (Fcorr - F0c) / (F0c + eps)
    if clip_min is not None:
        dff = np.maximum(dff, clip_min)
    return dff.astype(np.float32, copy=False), F0c.astype(np.float32, copy=False)

def compute_valid_masks(
    plane0_path: str | Path,
    # dF/F params
    win_s=60,
    pctl=20,
    eps=1e-5,
    F0_floor=1.0,
    clip_min=None,
    default_neucoeff=0.7,
    # validity thresholds
    neg_lo=-12.0,
    neg_hi=None,
    pos_lo=0.6,
    pos_hi=12.0,
):
    plane0 = Path(plane0_path)

    F, Fneu, ops, iscell = load_suite2p(plane0)
    r = float(ops.get("neucoeff", default_neucoeff))
    Fcorr = (np.asarray(F, dtype=np.float32) - r * np.asarray(Fneu, dtype=np.float32))

    F0 = compute_F0_percentile(Fcorr, ops, win_s=win_s, pctl=pctl)
    dff, _F0c = compute_dff(Fcorr, F0, eps=eps, F0_floor=F0_floor, clip_min=clip_min)

    good = load_good_frames(plane0)
    dff_use = dff[:, good] if good is not None else dff

    dmin = dff_use.min(axis=1)
    dmax = dff_use.max(axis=1)

    raw_roi = np.ones(dff.shape[0], dtype=bool)
    valid_roi = (dmax >= pos_lo) & (dmax <= pos_hi)
    iscell_mask = iscell.astype(bool)
    common = valid_roi & iscell_mask

    return {
        "raw_roi_mask": raw_roi,
        "valid_roi_mask": valid_roi.astype(bool),
        "iscell_mask": iscell_mask,
        "common_mask": common.astype(bool),
        "dmin": dmin.astype(np.float32),
        "dmax": dmax.astype(np.float32),
        "n_rois": int(dff.shape[0]),
        "n_cells_s2p": int(iscell_mask.sum()),
        "n_valid_rois": int(valid_roi.sum()),
        "n_common": int(common.sum()),
    }

def save_valid_outputs(plane0_path: str | Path, res: dict, overwrite: bool = False):
    plane0 = Path(plane0_path)

    out_raw    = plane0 / "raw_roi_mask.npy"
    out_valid  = plane0 / "valid_roi_mask.npy"
    out_iscell = plane0 / "iscell_mask.npy"
    out_common = plane0 / "common_mask.npy"
    out_stats  = plane0 / "valid_dmin_dmax.npz"

    if (not overwrite) and out_valid.exists() and out_common.exists():
        return

    np.save(out_raw, res["raw_roi_mask"])
    np.save(out_valid, res["valid_roi_mask"])
    np.save(out_iscell, res["iscell_mask"])
    np.save(out_common, res["common_mask"])
    np.savez_compressed(out_stats, dmin=res["dmin"], dmax=res["dmax"])