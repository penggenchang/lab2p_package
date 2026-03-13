from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from .masks import compute_valid_masks


def _load_good_frames_qc(plane0: Path):
    """
    Flexible good-frame loader for QC use.

    Checks several possible filenames. Returns:
      - boolean good-frame mask of shape (n_frames,), or
      - None if nothing is found
    """
    plane0 = Path(plane0)

    candidates = [
        plane0 / "good_frames.npy",
        plane0 / "bad_frames_mask.npy",
        plane0 / "bad_frames_final.npy",
        plane0.parent / "good_frames.npy",
    ]

    for p in candidates:
        if p.exists():
            arr = np.load(p)

            # already a boolean good-frame mask
            if arr.dtype == bool:
                # if this is actually a bad-frame mask file name, invert
                if "bad" in p.name.lower():
                    return ~arr.astype(bool).ravel()
                return arr.astype(bool).ravel()

            # integer indices: interpret as bad-frame indices
            if np.issubdtype(arr.dtype, np.integer):
                n_frames = np.load(plane0 / "spks.npy", mmap_mode="r").shape[1]
                good = np.ones(n_frames, dtype=bool)
                good[arr] = False
                return good

    return None


def load_selected_spks(
    plane0: Path,
    *,
    pos_lo: float = 0.8,
    pos_hi: float = 10.0,
    mask_key: str = "valid_roi_mask",
):
    """
    Load selected Suite2p deconvolved traces using the same ROI mask logic as lab2p QC.

    Returns
    -------
    X : ndarray
        shape (n_selected_rois, n_frames)
    roi_idx : ndarray
        original ROI indices
    res : dict
        compute_valid_masks result
    good : ndarray | None
        good-frame boolean mask if available
    """
    plane0 = Path(plane0)

    res = compute_valid_masks(plane0, pos_lo=pos_lo, pos_hi=pos_hi)
    mask = np.asarray(res[mask_key], dtype=bool).ravel()

    if mask.sum() == 0:
        raise RuntimeError(f"{mask_key} has 0 selected ROIs")

    spks_path = plane0 / "spks.npy"
    if not spks_path.exists():
        raise FileNotFoundError(f"Missing spks.npy: {spks_path}")

    spks = np.load(spks_path).astype(np.float32)

    good = _load_good_frames_qc(plane0)
    if good is not None:
        if len(good) != spks.shape[1]:
            raise ValueError(f"good_frames length {len(good)} != n_frames {spks.shape[1]}")
        spks = spks[:, good]

    X = spks[mask, :]
    roi_idx = np.flatnonzero(mask)

    return X, roi_idx, res, good


def compute_corr_matrix(X: np.ndarray):
    """
    Pearson correlation across ROIs.
    X shape: (n_rois, n_frames)
    """
    if X.shape[0] < 2:
        raise RuntimeError("Need at least 2 ROIs to compute a correlation matrix")
    if X.shape[1] < 2:
        raise RuntimeError("Need at least 2 frames to compute a correlation matrix")

    C = np.corrcoef(X)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    C = np.clip(C, -1.0, 1.0)
    np.fill_diagonal(C, 1.0)
    return C


def surrogate_analysis(
    traces,
    corr_matrix=None,
    n_surrogates=1000,
    method="random_shuffle",
    alpha=0.05,
    seed=0,
):
    """
    Surrogate significance test for correlation matrices.
    """
    rng = np.random.default_rng(seed)

    if corr_matrix is None:
        corr_matrix = np.corrcoef(traces)

    n_cells, n_timepoints = traces.shape
    exceed = np.zeros((n_cells, n_cells), dtype=np.int32)
    obs = np.abs(corr_matrix)

    for _ in range(n_surrogates):
        surrogate = np.empty_like(traces)

        for i in range(n_cells):
            if method == "time_shift":
                shift = rng.integers(0, n_timepoints)
                surrogate[i] = np.roll(traces[i], shift)
            elif method == "random_shuffle":
                surrogate[i] = rng.permutation(traces[i])
            else:
                raise ValueError("method must be 'time_shift' or 'random_shuffle'")

        surr_corr = np.corrcoef(surrogate)
        surr_corr = np.nan_to_num(surr_corr, nan=0.0, posinf=0.0, neginf=0.0)

        exceed += (np.abs(surr_corr) >= obs)

    p_values = (exceed + 1) / (n_surrogates + 1)
    np.fill_diagonal(p_values, 1.0)

    p_mask = p_values < alpha
    np.fill_diagonal(p_mask, False)

    robust_corr = corr_matrix.copy()
    robust_corr[~p_mask] = 0
    np.fill_diagonal(robust_corr, 1)

    return p_values, p_mask, robust_corr


def build_edge_table(roi_idx, corr_matrix, p_values=None, p_mask=None):
    """
    Build an edge list using the upper triangle only.
    """
    rows = []
    n = len(roi_idx)

    for i in range(n):
        for j in range(i + 1, n):
            rows.append({
                "matrix_i": i,
                "matrix_j": j,
                "roi_i": int(roi_idx[i]),
                "roi_j": int(roi_idx[j]),
                "pearson_r": float(corr_matrix[i, j]),
                "abs_r": float(abs(corr_matrix[i, j])),
                "p_value": float(p_values[i, j]) if p_values is not None else np.nan,
                "significant": bool(p_mask[i, j]) if p_mask is not None else False,
            })

    return pd.DataFrame(rows)


def export_network_qc_excel(
    plane0: Path,
    out_xlsx: Path,
    *,
    pos_lo: float = 0.8,
    pos_hi: float = 10.0,
    mask_key: str = "valid_roi_mask",
    do_surrogate: bool = True,
    n_surrogates: int = 1000,
    method: str = "random_shuffle",
    alpha: float = 0.05,
    seed: int = 0,
):
    """
    Export correlation/network QC results to Excel.

    Sheets:
      - pearson_r
      - p_values          (if do_surrogate)
      - p_mask            (if do_surrogate)
      - robust_corr       (if do_surrogate)
      - edges
      - roi_order
      - metadata
    """
    plane0 = Path(plane0)

    X, roi_idx, res, good = load_selected_spks(
        plane0,
        pos_lo=pos_lo,
        pos_hi=pos_hi,
        mask_key=mask_key,
    )

    corr_matrix = compute_corr_matrix(X)
    roi_labels = [f"ROI_{i}" for i in roi_idx]

    corr_df = pd.DataFrame(corr_matrix, index=roi_labels, columns=roi_labels)

    pval_df = None
    mask_df = None
    robust_df = None
    edges_df = None

    if do_surrogate:
        p_values, p_mask, robust_corr = surrogate_analysis(
            traces=X,
            corr_matrix=corr_matrix,
            n_surrogates=n_surrogates,
            method=method,
            alpha=alpha,
            seed=seed,
        )

        pval_df = pd.DataFrame(p_values, index=roi_labels, columns=roi_labels)
        mask_df = pd.DataFrame(p_mask.astype(int), index=roi_labels, columns=roi_labels)
        robust_df = pd.DataFrame(robust_corr, index=roi_labels, columns=roi_labels)
        edges_df = build_edge_table(roi_idx, corr_matrix, p_values=p_values, p_mask=p_mask)
    else:
        edges_df = build_edge_table(roi_idx, corr_matrix, p_values=None, p_mask=None)

    roi_df = pd.DataFrame({
        "matrix_order": np.arange(len(roi_idx)),
        "roi_index": roi_idx,
    })

    meta_df = pd.DataFrame({
        "TSeries": [plane0.parents[1].name],
        "plane0_path": [str(plane0)],
        "signal_type": ["suite2p_spks"],
        "pos_lo": [pos_lo],
        "pos_hi": [pos_hi],
        "mask_key": [mask_key],
        "n_selected_rois": [X.shape[0]],
        "n_frames": [X.shape[1]],
        "good_frames_applied": [good is not None],
        "do_surrogate": [do_surrogate],
        "surrogate_method": [method if do_surrogate else ""],
        "n_surrogates": [n_surrogates if do_surrogate else 0],
        "alpha": [alpha if do_surrogate else np.nan],
        "seed": [seed if do_surrogate else np.nan],
    })

    out_xlsx = Path(out_xlsx)
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        corr_df.to_excel(writer, sheet_name="pearson_r")
        roi_df.to_excel(writer, sheet_name="roi_order", index=False)
        edges_df.to_excel(writer, sheet_name="edges", index=False)
        meta_df.to_excel(writer, sheet_name="metadata", index=False)

        if do_surrogate:
            pval_df.to_excel(writer, sheet_name="p_values")
            mask_df.to_excel(writer, sheet_name="p_mask")
            robust_df.to_excel(writer, sheet_name="robust_corr")

    return out_xlsx


def find_plane0_dirs(proc_root: Path):
    proc_root = Path(proc_root)
    out = []
    for ts in sorted(proc_root.glob("TSeries*")):
        plane0 = ts / "suite2p" / "plane0"
        if (plane0 / "ops.npy").exists() and (plane0 / "spks.npy").exists():
            out.append(plane0)
    return out


def batch_export_network_qc(
    proc_root: Path,
    *,
    pos_lo: float = 0.8,
    pos_hi: float = 10.0,
    mask_key: str = "valid_roi_mask",
    do_surrogate: bool = True,
    n_surrogates: int = 1000,
    method: str = "random_shuffle",
    alpha: float = 0.05,
    seed: int = 0,
):
    """
    Batch export correlation/network QC Excel files into each TSeries/_QC_suite2p/.
    """
    proc_root = Path(proc_root)
    plane0_dirs = find_plane0_dirs(proc_root)

    counts = {"ok": 0, "fail": 0, "total": len(plane0_dirs)}
    tag = f"{str(pos_lo).replace('.', 'p')}_{str(pos_hi).replace('.', 'p')}"

    for plane0 in plane0_dirs:
        ts_name = plane0.parents[1].name
        qc_dir = plane0.parents[1] / "_QC_suite2p"
        qc_dir.mkdir(exist_ok=True)

        out_xlsx = qc_dir / f"{ts_name}__network_qc_{tag}.xlsx"

        try:
            export_network_qc_excel(
                plane0=plane0,
                out_xlsx=out_xlsx,
                pos_lo=pos_lo,
                pos_hi=pos_hi,
                mask_key=mask_key,
                do_surrogate=do_surrogate,
                n_surrogates=n_surrogates,
                method=method,
                alpha=alpha,
                seed=seed,
            )
            counts["ok"] += 1

        except Exception as e:
            fail_txt = qc_dir / "FAILED_network_qc.txt"
            fail_txt.write_text(f"{type(e).__name__}: {e}")
            counts["fail"] += 1

    return counts