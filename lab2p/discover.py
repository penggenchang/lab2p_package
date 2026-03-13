from __future__ import annotations
from pathlib import Path
import re

IMG_RE_DEFAULT = re.compile(r".*_Cycle\d+_Ch2_\d+\.ome\.tif{1,2}$", re.IGNORECASE)

def natural_key(s: str):
    import re as _re
    return [int(t) if t.isdigit() else t.lower() for t in _re.split(r"(\d+)", s)]

def find_tseries_dirs(raw_root: Path) -> list[Path]:
    raw_root = Path(raw_root)
    tseries = [p for p in raw_root.rglob("*") if p.is_dir() and p.name.startswith("TSeries")]
    return sorted(tseries, key=lambda p: natural_key(str(p)))

def list_image_tifs(tseries_dir: Path, img_re=IMG_RE_DEFAULT) -> list[Path]:
    tseries_dir = Path(tseries_dir)
    # only within the TSeries folder itself, ignore References/
    tifs = [p for p in tseries_dir.iterdir() if p.is_file() and img_re.match(p.name)]
    return sorted(tifs, key=lambda p: natural_key(p.name))