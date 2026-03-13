from __future__ import annotations
from pathlib import Path
from datetime import datetime
import json
import suite2p
from suite2p import run_s2p

from .discover import find_tseries_dirs, list_image_tifs
from .suite2p_settings import make_settings

def is_done(save_path0: Path) -> bool:
    return (Path(save_path0) / "suite2p" / "plane0" / "ops.npy").exists()

def out_path_for_tseries(out_root: Path, ts_dir: Path) -> Path:
    return Path(out_root) / ts_dir.name

def run_one(
    tseries_dir: Path,
    save_path0: Path,
    *,
    settings_kwargs: dict,
    skip_if_exists: bool = True,
) -> str:
    tseries_dir = Path(tseries_dir)
    save_path0 = Path(save_path0)

    if skip_if_exists and is_done(save_path0):
        return "skip_done"

    img_files = list_image_tifs(tseries_dir)
    if not img_files:
        return "skip_no_tifs"

    save_path0.mkdir(parents=True, exist_ok=True)

    settings = make_settings(**settings_kwargs)
    db = {
        "tiff_list": [str(p) for p in img_files],
        "data_path": [str(tseries_dir)],
        "save_path0": str(save_path0),
        "fast_disk": str(save_path0),
        "nplanes": 1,
        "nchannels": 1,
    }

    manifest = {
        "tseries_dir": str(tseries_dir),
        "image_files": [str(p) for p in img_files],
        "save_path0": str(save_path0),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "suite2p_path": getattr(suite2p, "__file__", None),
        "settings_summary": {
            "torch_device": settings.get("torch_device"),
            "fs": settings.get("fs"),
            "tau": settings.get("tau"),
            "diameter": settings.get("diameter").tolist()
                        if hasattr(settings.get("diameter"), "tolist")
                        else settings.get("diameter"),
            "detection_algorithm": settings.get("detection", {}).get("algorithm", None),
        },
    }
    (save_path0 / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    try:
        run_s2p(db=db, settings=settings)
        return "ok"
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        (save_path0 / "FAILED_exception.txt").write_text(msg)
        return "fail"

def batch_run(
    raw_root: Path,
    out_root: Path,
    *,
    settings_kwargs: dict,
    skip_if_exists: bool = True,
) -> dict:
    raw_root = Path(raw_root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    tseries_list = find_tseries_dirs(raw_root)
    if not tseries_list:
        raise RuntimeError(f"Found 0 TSeries folders under {raw_root}")

    counts = {"ok": 0, "fail": 0, "skip_done": 0, "skip_no_tifs": 0, "total": len(tseries_list)}

    for ts in tseries_list:
        outp = out_path_for_tseries(out_root, ts)
        status = run_one(ts, outp, settings_kwargs=settings_kwargs, skip_if_exists=skip_if_exists)
        counts[status] += 1

    return counts