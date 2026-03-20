from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

from .masks import compute_valid_masks


def summarize_fc_recording(
    plane0: Path,
    corr_excel: Path,
    pos_lo: float = 0.8,
    pos_hi: float = 10.0,
    mask_key: str = "common_mask",
):
    """
    Summarize functional connectivity metrics for one recording.
    """

    edges = pd.read_excel(corr_excel, sheet_name="edges")

    if edges.empty:
        raise RuntimeError(f"No edges found in {corr_excel}")

    # positive correlations only
    pos_edges = edges[edges["pearson_r"] > 0].copy()
    n_positive_fc = len(pos_edges)

    if n_positive_fc == 0:
        threshold = np.nan
        strong_edges = pos_edges.copy()
    else:
        threshold = float(pos_edges["pearson_r"].median())
        strong_edges = pos_edges[pos_edges["pearson_r"] >= threshold].copy()

    n_strong = len(strong_edges)

    # density calculation
    roi_nodes = pd.unique(edges[["roi_i", "roi_j"]].values.ravel())
    n_nodes = len(roi_nodes)
    possible_edges = n_nodes * (n_nodes - 1) / 2
    density = n_strong / possible_edges if possible_edges > 0 else np.nan

    median_strength = (
        float(strong_edges["pearson_r"].median()) if n_strong > 0 else np.nan
    )

    # ROI statistics
    res = compute_valid_masks(
        plane0,
        pos_lo=pos_lo,
        pos_hi=pos_hi,
    )

    total_rois = int(res["n_rois"])
    active_neurons = int(res["n_valid_rois"])

    dmax = np.asarray(res["dmax"], dtype=float)
    valid_mask = np.asarray(res["valid_roi_mask"], dtype=bool)

    median_peak_dff = float(np.median(dmax[valid_mask])) if valid_mask.any() else np.nan

    return {
        "threshold_median_fc": threshold,
        "n_positive_pearson_validate": n_positive_fc,
        "n_fc_above_median": n_strong,
        "strong_fc_density_over_all": density,
        "median_strong_fc_strength": median_strength,
        "Total_rois": total_rois,
        "Active_neurons": active_neurons,
        "median_peak_dff": median_peak_dff,
    }


def summarize_fc_group(
    proc_root: Path,
    *,
    pos_lo: float = 0.8,
    pos_hi: float = 10.0,
    mask_key: str = "common_mask",
    out_name: str = "fc_summary.xlsx",
):
    """
    Summarize functional connectivity across all TSeries.

    Requires:
        TSeries-*/_QC_suite2p/*network_qc*.xlsx
    """

    proc_root = Path(proc_root)
    rows = []

    for ts in sorted(proc_root.glob("TSeries*")):

        plane0 = ts / "suite2p" / "plane0"
        qc_dir = ts / "_QC_suite2p"

        if not plane0.exists() or not qc_dir.exists():
            continue

        corr_files = sorted(qc_dir.glob("*network_qc*.xlsx"))
        if not corr_files:
            continue

        corr_excel = corr_files[0]

        metrics = summarize_fc_recording(
            plane0=plane0,
            corr_excel=corr_excel,
            pos_lo=pos_lo,
            pos_hi=pos_hi,
            mask_key=mask_key,
        )

        rows.append({
            "recording": ts.name,
            "corr_excel": str(corr_excel),
            **metrics,
        })

    if not rows:
        raise RuntimeError(
            f"No usable recordings found under {proc_root}.\n"
            f"Expected files like:\n"
            f"  TSeries-*/_QC_suite2p/*network_qc*.xlsx"
        )

    df = pd.DataFrame(rows).sort_values("recording").reset_index(drop=True)

    #out_dir = proc_root / "_summary"
    #out_dir.mkdir(exist_ok=True)
    out_dir = proc_root

    out_path = out_dir / out_name
    df.to_excel(out_path, index=False)

    return out_path