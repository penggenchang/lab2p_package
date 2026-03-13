from __future__ import annotations
import numpy as np
import suite2p

def ensure(d, *keys):
    cur = d
    for k in keys:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    return cur

def _looks_like_settings(obj):
    return isinstance(obj, dict) and ("registration" in obj or "run" in obj or "io" in obj)

def get_suite2p_defaults(verbose: bool = False) -> dict:
    """
    Robustly find a callable in suite2p / suite2p.default_ops / suite2p.ops
    that returns a settings dict (Suite2p 1.0.0.1 friendly).
    """
    candidates = []
    mods = []
    for modname in ("default_ops", "ops"):
        m = getattr(suite2p, modname, None)
        if m is not None:
            mods.append((modname, m))
    mods.append(("suite2p", suite2p))

    for modname, m in mods:
        for name, obj in vars(m).items():
            if callable(obj) and name.lower().startswith(("default", "settings")):
                candidates.append((modname, name, obj))

    for modname, name, fn in candidates:
        try:
            out = fn()
            if _looks_like_settings(out):
                if verbose:
                    print(f"[defaults] using {modname}.{name}()")
                return out
        except Exception:
            pass

    raise RuntimeError(
        "Could not locate a defaults function returning a settings dict in this Suite2p install.\n"
        f"suite2p loaded from: {getattr(suite2p, '__file__', 'UNKNOWN')}"
    )

def make_settings(
    *,
    torch_device: str = "cpu",     # "cpu" or "cuda"
    fs: float = 1000/134.92,
    tau: float = 1.25,
    diameter=(12.0, 12.0),
    th_badframes: float = 0.7,
) -> dict:
    """
    Suite2p 1.0.0.1-safe settings builder:
      - start from install defaults
      - ensure required nested keys exist
      - apply your lab's stable overrides
    """
    s = get_suite2p_defaults(verbose=False)

    ensure(s, "run")
    ensure(s, "io")
    ensure(s, "registration")
    ensure(s, "detection")
    ensure(s, "extraction")
    ensure(s, "classification")

    # top-level
    s["torch_device"] = torch_device
    s["fs"] = float(fs)
    s["tau"] = float(tau)
    s["diameter"] = np.array(diameter, dtype=float)

    # run flags
    run = s["run"]
    run.setdefault("do_registration", 1)
    run.setdefault("do_regmetrics", True)
    run.setdefault("do_detection", True)
    run.setdefault("do_deconvolution", True)
    run.setdefault("multiplane_parallel", False)

    # io expected keys
    io = s["io"]
    io.setdefault("combined", True)
    io.setdefault("save_mat", False)
    io.setdefault("save_NWB", False)
    io.setdefault("save_ops_orig", True)
    io.setdefault("delete_bin", False)
    io.setdefault("move_bin", False)

    # registration expected keys
    reg = s["registration"]
    reg.setdefault("align_by_chan2", False)
    reg.setdefault("reg_tif", False)
    reg.setdefault("reg_tif_chan2", False)
    reg.setdefault("do_bidiphase", False)
    reg.setdefault("bidiphase", 0.0)
    reg.setdefault("nimg_init", 400)
    reg.setdefault("batch_size", 100)
    reg.setdefault("smooth_sigma_time", 0)
    reg.setdefault("smooth_sigma", 1.15)
    reg.setdefault("spatial_taper", 3.45)
    reg.setdefault("subpixel", 10)
    reg.setdefault("two_step_registration", False)

    # lab overrides
    reg["nonrigid"] = True
    reg["block_size"] = (128, 128)
    reg["maxregshift"] = 0.1
    reg["maxregshiftNR"] = 5
    reg["snr_thresh"] = 1.2
    reg["align_by_chan2"] = False
    reg["th_badframes"] = float(th_badframes)
    reg["norm_frames"] = True

    # detection defaults (sourcery)
    det = s["detection"]
    det.setdefault("algorithm", "sourcery")
    det.setdefault("nbins", 5000)
    det.setdefault("block_size", (64, 64))
    det.setdefault("denoise", False)
    det.setdefault("bin_size", None)
    det.setdefault("highpass_time", 100)
    det.setdefault("threshold_scaling", 1.0)
    det.setdefault("npix_norm_min", 0.0)
    det.setdefault("npix_norm_max", 100.0)
    det.setdefault("max_overlap", 0.75)
    det.setdefault("soma_crop", True)
    det.setdefault("chan2_threshold", 0.25)
    det.setdefault("cellpose_chan2", False)
    det.setdefault("sourcery_settings", {})
    det["sourcery_settings"].setdefault("connected", True)
    det["sourcery_settings"].setdefault("max_iterations", 20)
    det["sourcery_settings"].setdefault("smooth_masks", False)
    det["algorithm"] = "sourcery"

    # extraction expected keys
    ext = s["extraction"]
    ext.setdefault("snr_threshold", 0.0)
    ext.setdefault("batch_size", 500)
    ext.setdefault("neuropil_extract", True)
    ext.setdefault("neuropil_coefficient", 0.7)
    ext.setdefault("inner_neuropil_radius", 2)
    ext.setdefault("min_neuropil_pixels", 350)
    ext.setdefault("lam_percentile", 50.0)
    ext.setdefault("allow_overlap", False)
    ext.setdefault("circular_neuropil", False)

    # classification expected keys
    cls = s["classification"]
    cls.setdefault("use_builtin_classifier", True)
    cls.setdefault("classifier_path", None)
    cls.setdefault("preclassify", 0.0)

    return s