from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from .masks import (
    compute_valid_masks,
    load_suite2p,
    load_good_frames,
    compute_F0_percentile,
    compute_dff,
)


def fmt_range(lo, hi):
    return f"{str(lo).replace('.', 'p')}_{str(hi).replace('.', 'p')}"


def zscore_rows(X):
    mu = X.mean(axis=1, keepdims=True)
    sigma = X.std(axis=1, keepdims=True) + 1e-5
    return (X - mu) / sigma


def export_dff_excel(
    plane0: Path,
    out_xlsx: Path,
    *,
    pos_lo: float = 0.8,
    pos_hi: float = 8.0,
    mask_key: str = "valid_roi_mask",
    win_s: float = 60,
    pctl: float = 20,
    eps: float = 1e-5,
    F0_floor: float = 1.0,
    clip_min=None,
):
    """
    Export dF/F traces for selected ROIs to Excel.

    Sheets:
      - dff_traces
      - roi_order
      - metadata
    """
    plane0 = Path(plane0)

    res = compute_valid_masks(plane0, pos_lo=pos_lo, pos_hi=pos_hi)
    mask = np.asarray(res[mask_key], dtype=bool).ravel()
    if mask.sum() == 0:
        raise RuntimeError(f"{mask_key} has 0 cells")

    F, Fneu, ops, _iscell = load_suite2p(plane0)
    r = float(ops.get("neucoeff", 0.7))
    Fcorr = np.asarray(F, dtype=np.float32) - r * np.asarray(Fneu, dtype=np.float32)

    F0 = compute_F0_percentile(Fcorr, ops, win_s=win_s, pctl=pctl)
    dff, _ = compute_dff(Fcorr, F0, eps=eps, F0_floor=F0_floor, clip_min=clip_min)

    good = load_good_frames(plane0)
    if good is not None:
        dff = dff[:, good]

    selected_roi_idx = np.flatnonzero(mask)
    X = dff[mask, :]

    # same sorting as heatmap
    order = np.argsort(X.sum(axis=1))[::-1]
    X = X[order, :]
    selected_roi_idx = selected_roi_idx[order]

    df = pd.DataFrame(
        X.T,
        columns=[f"ROI_{roi}" for roi in selected_roi_idx]
    )
    df.insert(0, "frame", np.arange(X.shape[1]))

    roi_df = pd.DataFrame({
        "sorted_export_order": np.arange(len(selected_roi_idx)),
        "roi_index": selected_roi_idx,
    })

    meta_df = pd.DataFrame({
        "TSeries": [plane0.parents[1].name],
        "plane0_path": [str(plane0)],
        "signal_type": ["dff"],
        "mask_key": [mask_key],
        "pos_lo": [pos_lo],
        "pos_hi": [pos_hi],
        "n_selected_rois": [len(selected_roi_idx)],
        "n_frames": [X.shape[1]],
        "neucoeff": [r],
        "good_frames_applied": [good is not None],
    })

    out_xlsx = Path(out_xlsx)
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="dff_traces", index=False)
        roi_df.to_excel(writer, sheet_name="roi_order", index=False)
        meta_df.to_excel(writer, sheet_name="metadata", index=False)

    return out_xlsx


def export_spks_normalized_excel(
    plane0: Path,
    out_xlsx: Path,
    *,
    pos_lo: float = 0.8,
    pos_hi: float = 8.0,
    mask_key: str = "valid_roi_mask",
):
    """
    Export z-scored Suite2p deconvolved traces (spks.npy) for selected ROIs to Excel.

    Sheets:
      - spks_traces
      - roi_order
      - metadata
    """
    plane0 = Path(plane0)

    res = compute_valid_masks(plane0, pos_lo=pos_lo, pos_hi=pos_hi)
    mask = np.asarray(res[mask_key], dtype=bool).ravel()
    if mask.sum() == 0:
        raise RuntimeError(f"{mask_key} has 0 cells")

    spks_path = plane0 / "spks.npy"
    if not spks_path.exists():
        raise FileNotFoundError(f"Missing spks.npy: {spks_path}")

    spks = np.load(spks_path).astype(np.float32)

    good = load_good_frames(plane0)
    if good is not None:
        spks = spks[:, good]

    selected_roi_idx = np.flatnonzero(mask)
    X = spks[mask, :]
    Xz = zscore_rows(X)

    # same sorting as heatmap/export: by raw spks sum
    order = np.argsort(X.sum(axis=1))[::-1]
    X = X[order, :]
    Xz = Xz[order, :]
    selected_roi_idx = selected_roi_idx[order]

    df = pd.DataFrame(
        Xz.T,
        columns=[f"ROI_{roi}" for roi in selected_roi_idx]
    )
    df.insert(0, "frame", np.arange(X.shape[1]))

    roi_df = pd.DataFrame({
        "sorted_export_order": np.arange(len(selected_roi_idx)),
        "roi_index": selected_roi_idx,
    })

    meta_df = pd.DataFrame({
        "TSeries": [plane0.parents[1].name],
        "plane0_path": [str(plane0)],
        "signal_type": ["suite2p_spks_zscore"],
        "mask_key": [mask_key],
        "pos_lo": [pos_lo],
        "pos_hi": [pos_hi],
        "n_selected_rois": [len(selected_roi_idx)],
        "n_frames": [X.shape[1]],
        "good_frames_applied": [good is not None],
    })

    out_xlsx = Path(out_xlsx)
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="spks_traces", index=False)
        roi_df.to_excel(writer, sheet_name="roi_order", index=False)
        meta_df.to_excel(writer, sheet_name="metadata", index=False)

    return out_xlsx


def find_plane0_dirs(proc_root: Path):
    proc_root = Path(proc_root)
    out = []
    for ts in sorted(proc_root.glob("TSeries*")):
        plane0 = ts / "suite2p" / "plane0"
        if (plane0 / "ops.npy").exists():
            out.append(plane0)
    return out


def batch_export_traces_excel(
    proc_root: Path,
    *,
    pos_lo: float = 0.8,
    pos_hi: float = 8.0,
    mask_key: str = "valid_roi_mask",
    export_dff: bool = True,
    export_spks: bool = True,
):
    """
    Batch export dF/F and z-scored deconvolved traces to Excel.
    Saves files into each TSeries/_QC_suite2p/.
    """
    proc_root = Path(proc_root)
    plane0_dirs = find_plane0_dirs(proc_root)

    counts = {"ok": 0, "fail": 0, "total": len(plane0_dirs)}
    tag = fmt_range(pos_lo, pos_hi)

    for plane0 in plane0_dirs:
        ts_name = plane0.parents[1].name
        qc_dir = plane0.parents[1] / "_QC_suite2p"
        qc_dir.mkdir(exist_ok=True)

        try:
            if export_dff:
                out_dff = qc_dir / f"{ts_name}__dff_{tag}.xlsx"
                export_dff_excel(
                    plane0,
                    out_dff,
                    pos_lo=pos_lo,
                    pos_hi=pos_hi,
                    mask_key=mask_key,
                )

            if export_spks:
                out_spks = qc_dir / f"{ts_name}__deconvolved_{tag}.xlsx"
                export_spks_normalized_excel(
                    plane0,
                    out_spks,
                    pos_lo=pos_lo,
                    pos_hi=pos_hi,
                    mask_key=mask_key,
                )

            counts["ok"] += 1

        except Exception as e:
            fail_txt = qc_dir / "FAILED_trace_export.txt"
            fail_txt.write_text(f"{type(e).__name__}: {e}")
            counts["fail"] += 1

    return counts