from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

from .masks import (
    compute_valid_masks,
    load_suite2p,
    load_good_frames,
    compute_F0_percentile,
    compute_dff,
)

def save_heatmap_common(
    plane0: Path,
    out_svg: Path,
    *,
    pos_lo: float,
    pos_hi: float,
    mask_key: str = "common_mask",
    clip_for_display: bool = True,
    vmin_pct: float = 1,
    vmax_pct: float = 99,
    dpi: int = 300,
):
    res = compute_valid_masks(plane0, pos_lo=pos_lo, pos_hi=pos_hi)
    mask = np.asarray(res[mask_key], dtype=bool).ravel()
    if mask.sum() == 0:
        raise RuntimeError(f"{mask_key} has 0 cells")

    F, Fneu, ops, _ = load_suite2p(plane0)
    r = float(ops.get("neucoeff", 0.7))
    Fcorr = (np.asarray(F, np.float32) - r * np.asarray(Fneu, np.float32))

    F0 = compute_F0_percentile(Fcorr, ops, win_s=60, pctl=20)
    dff, _ = compute_dff(Fcorr, F0)

    good = load_good_frames(plane0)
    if good is not None:
        dff = dff[:, good]

    X = dff[mask, :]
    Xp = np.clip(X, pos_lo, pos_hi) if clip_for_display else X

    order = np.argsort(Xp.sum(axis=1))[::-1]
    Xp = Xp[order, :]

    vmin = np.percentile(Xp, vmin_pct)
    vmax = np.percentile(Xp, vmax_pct)

    ts_name = Path(plane0).parents[1].name
    t = np.arange(Xp.shape[1])

    fig, ax = plt.subplots(figsize=(4.6, 3.4), dpi=dpi, layout="constrained")
    im = ax.imshow(
        Xp, aspect="auto", origin="lower", interpolation="nearest",
        cmap="viridis", vmin=vmin, vmax=vmax,
        extent=[t[0], t[-1], 0, Xp.shape[0]]
    )
    ax.set_title(f"{ts_name} | heatmap({pos_lo}–{pos_hi})")
    ax.set_xlabel("Frames")
    ax.set_ylabel("Cell index (sorted)")
    plt.colorbar(im, ax=ax).set_label("dF/F (display)")

    out_svg = Path(out_svg)
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)

def save_roi_circles_one_range(
    plane0: Path,
    out_svg: Path,
    *,
    pos_lo: float,
    pos_hi: float,
    mask_key: str = "common_mask",
    circle_r: float = 2.8,
    edgecolor: str = "yellow",
    lw: float = 0.9,
    alpha: float = 0.9,
    sort_like_heatmap: bool = True,
    label_mode: str = "row",   # "row" | "roi" | "none"
    label_fontsize: int = 6,
    dpi: int = 300,
):
    plane0 = Path(plane0)

    res = compute_valid_masks(plane0, pos_lo=pos_lo, pos_hi=pos_hi)
    mask = np.asarray(res[mask_key], dtype=bool).ravel()
    if mask.sum() == 0:
        raise RuntimeError(f"{mask_key} has 0 cells for range {pos_lo}–{pos_hi}")

    ops = np.load(plane0 / "ops.npy", allow_pickle=True).item()
    stat = np.load(plane0 / "stat.npy", allow_pickle=True)
    mean_img = ops.get("meanImg", None)
    if mean_img is None:
        raise RuntimeError("ops['meanImg'] missing")

    n_stat = len(stat)
    if mask.shape[0] != n_stat:
        mask = mask[:n_stat]

    idx0 = np.where(mask)[0]
    if idx0.size == 0:
        raise RuntimeError(f"{mask_key} selected 0 cells after length reconcile")

    idx_plot = idx0
    if sort_like_heatmap:
        F, Fneu, ops2, _ = load_suite2p(plane0)
        r = float(ops2.get("neucoeff", 0.7))
        Fcorr = (np.asarray(F, np.float32) - r * np.asarray(Fneu, np.float32))

        F0 = compute_F0_percentile(Fcorr, ops2, win_s=60, pctl=20)
        dff, _ = compute_dff(Fcorr, F0)

        good = load_good_frames(plane0)
        if good is not None:
            dff = dff[:, good]

        X = dff[idx0, :]
        Xp = np.clip(X, pos_lo, pos_hi)
        order = np.argsort(Xp.sum(axis=1))[::-1]
        idx_plot = idx0[order]

    ts_name = plane0.parents[1].name
    fig, ax = plt.subplots(figsize=(7, 7), dpi=dpi, layout="constrained")
    ax.imshow(mean_img, cmap="gray")
    ax.axis("off")
    ax.set_title(f"{ts_name} | ROI map({pos_lo}–{pos_hi}) | {mask_key} n={len(idx_plot)}")

    for row, roi_i in enumerate(idx_plot):
        s = stat[int(roi_i)]
        yc = float(np.mean(s["ypix"]))
        xc = float(np.mean(s["xpix"]))

        ax.add_patch(Circle((xc, yc), circle_r, edgecolor=edgecolor, facecolor="none", linewidth=lw, alpha=alpha))

        if label_mode == "none":
            continue
        lab = str(row) if label_mode == "row" else str(int(roi_i))
        ax.text(
            xc + 2, yc - 2, lab,
            color="red", fontsize=label_fontsize, weight="bold",
            ha="center", va="center",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.5),
        )

    out_svg = Path(out_svg)
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)

def save_roi_circles_two_ranges(
    plane0: Path,
    out_svg: Path,
    *,
    posA=(0.6, 12.0),
    posB=(0.8, 8.0),
    mask_key="common_mask",
    circle_r=2.8,
    colorA="yellow",
    colorB="cyan",
    show_overlap=True,
    overlap_color="lime",
    lwA=0.9,
    lwB=0.9,
    lwOverlap=1.8,
    alphaA=0.85,
    alphaB=0.85,
    alphaOverlap=0.95,
    label_mode="overlap_only",  # "none" | "roi" | "overlap_only"
    label_fontsize=6,
    dpi=300,
):
    ops = np.load(Path(plane0) / "ops.npy", allow_pickle=True).item()
    stat = np.load(Path(plane0) / "stat.npy", allow_pickle=True)
    mean_img = ops.get("meanImg", None)
    if mean_img is None:
        raise RuntimeError("ops['meanImg'] missing")
    n_stat = len(stat)

    loA, hiA = posA
    loB, hiB = posB
    resA = compute_valid_masks(plane0, pos_lo=loA, pos_hi=hiA)
    resB = compute_valid_masks(plane0, pos_lo=loB, pos_hi=hiB)

    maskA = np.asarray(resA[mask_key], dtype=bool).ravel()[:n_stat]
    maskB = np.asarray(resB[mask_key], dtype=bool).ravel()[:n_stat]

    idxA = np.where(maskA)[0]
    idxB = np.where(maskB)[0]
    overlap = sorted(set(idxA.tolist()) & set(idxB.tolist()))
    overlap_set = set(overlap)

    ts_name = Path(plane0).parents[1].name
    fig, ax = plt.subplots(figsize=(7, 7), dpi=dpi, layout="constrained")
    ax.imshow(mean_img, cmap="gray")
    ax.axis("off")

    ax.set_title(f"{ts_name} | {mask_key}", pad=10)
    ax.text(0.02, 1.02, f"A ({loA}–{hiA}) n={len(idxA)}", transform=ax.transAxes,
            color=colorA, fontsize=10, fontweight="bold", ha="left", va="bottom")
    ax.text(0.42, 1.02, f"B ({loB}–{hiB}) n={len(idxB)}", transform=ax.transAxes,
            color=colorB, fontsize=10, fontweight="bold", ha="left", va="bottom")
    if show_overlap:
        ax.text(0.75, 1.02, f"overlap n={len(overlap)}", transform=ax.transAxes,
                color=overlap_color, fontsize=10, fontweight="bold", ha="left", va="bottom")

    def draw(idxs, edgecolor, lw, alpha):
        for roi_i in idxs:
            s = stat[int(roi_i)]
            yc = float(np.mean(s["ypix"]))
            xc = float(np.mean(s["xpix"]))
            ax.add_patch(Circle((xc, yc), circle_r, edgecolor=edgecolor, facecolor="none", linewidth=lw, alpha=alpha))

            if label_mode == "roi":
                do_label = True
            elif label_mode == "overlap_only":
                do_label = (roi_i in overlap_set)
            else:
                do_label = False

            if do_label:
                ax.text(
                    xc + 2, yc - 2, str(int(roi_i)), color="red",
                    fontsize=label_fontsize, weight="bold",
                    ha="center", va="center",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.5),
                )

    draw(idxA, colorA, lwA, alphaA)
    draw(idxB, colorB, lwB, alphaB)
    if show_overlap and overlap:
        draw(overlap, overlap_color, lwOverlap, alphaOverlap)

    handles = [
        Line2D([0], [0], color=colorA, lw=lwA, label=f"A ({loA}–{hiA})  n={len(idxA)}"),
        Line2D([0], [0], color=colorB, lw=lwB, label=f"B ({loB}–{hiB})  n={len(idxB)}"),
    ]
    if show_overlap:
        handles.append(Line2D([0], [0], color=overlap_color, lw=lwOverlap, label=f"Overlap  n={len(overlap)}"))
    ax.legend(handles=handles, loc="upper right", frameon=True, fontsize=9)

    out_svg = Path(out_svg)
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)