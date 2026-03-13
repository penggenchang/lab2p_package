from __future__ import annotations
from pathlib import Path
import pandas as pd

from .run_suite2p import batch_run
from .masks import compute_valid_masks, save_valid_outputs
from .qc_plots import (
    save_heatmap_common,
    save_roi_circles_one_range,
    save_roi_circles_two_ranges,
)

def find_plane0_dirs(out_root: Path) -> list[Path]:
    out = []
    for ts in sorted(Path(out_root).glob("TSeries*")):
        plane0 = ts / "suite2p" / "plane0"
        if (plane0 / "ops.npy").exists() and (plane0 / "stat.npy").exists():
            out.append(plane0)
    return out

def run_qc_pipeline(
    *,
    raw_root: Path,
    out_root: Path,
    torch_device: str = "cpu",
    fs: float = 1000/134.92,
    tau: float = 1.25,
    diameter=(12.0, 12.0),
    th_badframes: float = 0.7,
    # single-range QC choice
    pos_lo: float = 0.8,
    pos_hi: float = 8.0,
    mask_key: str = "common_mask",
    # optional compare plot
    compare_ranges=((0.6, 12.0), (0.8, 8.0)),
    do_compare: bool = True,
    skip_suite2p_if_exists: bool = True,
    dpi: int = 300,
) -> Path:
    raw_root = Path(raw_root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # 1) Suite2p batch (skip if already processed)
    counts = batch_run(
        raw_root=raw_root,
        out_root=out_root,
        settings_kwargs=dict(
            torch_device=torch_device,
            fs=fs,
            tau=tau,
            diameter=diameter,
            th_badframes=th_badframes,
        ),
        skip_if_exists=skip_suite2p_if_exists,
    )
    print("[Suite2p batch]", counts)

    # 2) QC per TSeries
    plane0_dirs = find_plane0_dirs(out_root)
    print(f"[QC] Found {len(plane0_dirs)} plane0 dirs under {out_root}")

    rows = []
    ok = fail = 0

    for plane0 in plane0_dirs:
        ts_dir = plane0.parents[1]
        ts_name = ts_dir.name
        qc_dir = ts_dir / "_QC_suite2p"
        qc_dir.mkdir(exist_ok=True)

        # Save masks next to plane0
        try:
            res = compute_valid_masks(plane0, pos_lo=pos_lo, pos_hi=pos_hi)
            save_valid_outputs(plane0, res, overwrite=False)
        except Exception as e:
            (qc_dir / "FAILED_masks.txt").write_text(f"{type(e).__name__}: {e}")

        heatmap_path = qc_dir / f"{ts_name}__heatmap_{pos_lo}_{pos_hi}.svg"
        roi_one_path = qc_dir / f"{ts_name}__roi_{pos_lo}_{pos_hi}.svg"
        roi_cmp_path = qc_dir / f"{ts_name}__roi_compare_{compare_ranges[0]}__vs__{compare_ranges[1]}.svg"

        # make filenames Windows-safe (replace '.' with 'p')
        heatmap_path = Path(str(heatmap_path).replace(".", "p"))
        roi_one_path = Path(str(roi_one_path).replace(".", "p"))
        roi_cmp_path = Path(str(roi_cmp_path).replace(".", "p"))

        try:
            save_heatmap_common(
                plane0, heatmap_path,
                pos_lo=pos_lo, pos_hi=pos_hi,
                mask_key=mask_key,
                dpi=dpi,
            )
            save_roi_circles_one_range(
                plane0, roi_one_path,
                pos_lo=pos_lo, pos_hi=pos_hi,
                mask_key=mask_key,
                sort_like_heatmap=True,
                label_mode="row",
                dpi=dpi,
            )
            if do_compare:
                save_roi_circles_two_ranges(
                    plane0, roi_cmp_path,
                    posA=compare_ranges[0],
                    posB=compare_ranges[1],
                    mask_key=mask_key,
                    label_mode="overlap_only",
                    show_overlap=True,
                    dpi=dpi,
                )
            ok += 1
        except Exception as e:
            (qc_dir / "FAILED_qc.txt").write_text(f"{type(e).__name__}: {e}")
            fail += 1

        rows.append({
            "TSeries": ts_name,
            "plane0": str(plane0),
            "qc_dir": str(qc_dir),
            "heatmap_path": str(heatmap_path),
            "roi_one_path": str(roi_one_path),
            "roi_compare_path": str(roi_cmp_path) if do_compare else "",
        })

    print(f"[QC] Done. ok={ok} fail={fail}")

    df = pd.DataFrame(rows).sort_values("TSeries").reset_index(drop=True)
    record_xlsx = out_root / "QC_finish.xlsx"
    df.to_excel(record_xlsx, index=False)
    print("[QC] finish excel:", record_xlsx)
    ## 0306: summary of ROI information
    from .summary import summarize_rois
    summarize_rois(out_root, pos_lo=pos_lo, pos_hi=pos_hi)


    return record_xlsx

