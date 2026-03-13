from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from .masks import compute_valid_masks


def find_plane0_dirs(proc_root: Path):
    """
    Find all Suite2p plane0 folders inside processed directory.
    """
    proc_root = Path(proc_root)

    out = []
    for ts in sorted(proc_root.glob("TSeries*")):
        plane0 = ts / "suite2p" / "plane0"
        if (plane0 / "ops.npy").exists():
            out.append(plane0)

    return out


def _safe_mean(x):
    x = np.asarray(x)
    return float(np.mean(x)) if x.size > 0 else np.nan


def _safe_median(x):
    x = np.asarray(x)
    return float(np.median(x)) if x.size > 0 else np.nan


def summarize_rois(
    proc_root: Path,
    *,
    pos_lo: float = 0.8,
    pos_hi: float = 8.0,
    out_name: str = "ROI_summary.xlsx",
):
    """
    Summarize ROI counts and simple recording-level activity metrics.

    Output columns:
        TSeries
        n_rois
        n_cells_s2p
        n_valid_rois
        n_common
        frac_valid
        frac_common
        mean_dmax_valid
        median_dmax_valid
        mean_dmax_common
        median_dmax_common
        mean_dmin_valid
        median_dmin_valid
    """
    proc_root = Path(proc_root)
    plane0_dirs = find_plane0_dirs(proc_root)

    rows = []

    for plane0 in plane0_dirs:
        ts_name = plane0.parents[1].name

        res = compute_valid_masks(
            plane0,
            pos_lo=pos_lo,
            pos_hi=pos_hi,
        )

        valid_mask = np.asarray(res["valid_roi_mask"], dtype=bool)
        common_mask = np.asarray(res["common_mask"], dtype=bool)
        dmax = np.asarray(res["dmax"], dtype=float)
        dmin = np.asarray(res["dmin"], dtype=float)

        dmax_valid = dmax[valid_mask]
        dmax_common = dmax[common_mask]
        dmin_valid = dmin[valid_mask]

        rows.append(
            {
                "TSeries": ts_name,
                "n_rois": res["n_rois"],
                "n_cells_s2p": res["n_cells_s2p"],
                "n_valid_rois": res["n_valid_rois"],
                "n_common": res["n_common"],

                "ratio_valid": res["n_valid_rois"] / res["n_rois"] if res["n_rois"] > 0 else np.nan,
                "ratio_common": res["n_common"] / res["n_rois"] if res["n_rois"] > 0 else np.nan,

                "mean_dff_peak_valid": _safe_mean(dmax_valid),
                "median_dff_peak_valid": _safe_median(dmax_valid),

                "mean_dff_peak_common": _safe_mean(dmax_common),
                "median_dff_peak_common": _safe_median(dmax_common),
                
               # "mean_dmin_valid": _safe_mean(dmin_valid),
               # "median_dmin_valid": _safe_median(dmin_valid),
            }
        )

    df = pd.DataFrame(rows).sort_values("TSeries").reset_index(drop=True)

    out_path = proc_root / out_name
    df.to_excel(out_path, index=False)
    print('ROI numbers and peak dFF:', out_path)

    return out_path