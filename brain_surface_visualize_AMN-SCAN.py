#!/usr/bin/env python3
"""Single-map AMN-SCAN surface renderer for one 3D MNI152 NIfTI image."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np


HERE = Path(__file__).resolve().parent
ENGINE_FILE = HERE / "03_4_visualize_AMN_FIR-DG-FIR-DG_LRResponse_dualWM_fs6_mne.py"
PROFILES = {"fsaverage5": ("10k", 10242), "fsaverage6": ("41k", 40962)}
DEFAULT_FS5 = Path("/software/Tools/freesurfer/subjects/fsaverage5")
DEFAULT_FS6 = Path("/software/Tools/freesurfer/subjects/fsaverage6")
DEFAULT_SCAN = Path("/home/wangxr/.cache/templateflow/tpl-Altas/SCAN_atlas_OSF_nmdka")
DEFAULT_AMN = Path("/home/wangxr/.cache/templateflow/tpl-Altas/AMN_Dosenbach")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file", type=Path, help="One 3D MNI152 NIfTI (.nii or .nii.gz)")
    parser.add_argument("output_dir", type=Path, help="Directory for the PNG, projections, and reports")
    parser.add_argument("output_basename", help="Filename prefix without a path separator")
    parser.add_argument("--surface-space", choices=tuple(PROFILES), default="fsaverage6")
    parser.add_argument("--surface-dir", type=Path, default=None)
    parser.add_argument("--fsaverage5-dir", type=Path, default=DEFAULT_FS5)
    parser.add_argument("--fsaverage6-dir", type=Path, default=DEFAULT_FS6)
    parser.add_argument("--surface-mix-pial", type=float, default=0.4)
    parser.add_argument("--surface-mix-inflated", type=float, default=0.6)
    parser.add_argument("--amn-left-label", type=Path, default=DEFAULT_AMN / "lh_AMN2025_YeoVAN-PowerCON_union_fsaverage5.label.gii")
    parser.add_argument("--amn-right-label", type=Path, default=DEFAULT_AMN / "rh_AMN2025_YeoVAN-PowerCON_union_fsaverage5.label.gii")
    parser.add_argument("--amn-reference-space", choices=("fsaverage5", "target"), default="fsaverage5")
    parser.add_argument("--scan-atlas-dir", type=Path, default=DEFAULT_SCAN)
    parser.add_argument("--scan-left-label", type=Path, default=None)
    parser.add_argument("--scan-right-label", type=Path, default=None)
    parser.add_argument("--scan-label-value", type=float, default=1.5)
    parser.add_argument("--mode", choices=("activation", "fc"), default="activation")
    parser.add_argument("--lut-file", type=Path, default=None)
    parser.add_argument("--display-min", type=float, default=None)
    parser.add_argument("--display-mid", type=float, default=None)
    parser.add_argument("--display-max", type=float, default=None)
    parser.add_argument("--contour-width", type=float, default=26.0)
    parser.add_argument("--boundary-offset", type=float, default=0.55)
    parser.add_argument("--boundary-inset", type=float, default=0.80)
    parser.add_argument("--amn-collapse-scope", choices=("all", "near-scan"), default="all")
    parser.add_argument("--amn-extra-inset", type=float, default=0.05)
    parser.add_argument("--amn-scan-contact-mm", type=float, default=4.0)
    parser.add_argument("--amn-fragment-max-vertices", type=int, default=3)
    parser.add_argument("--amn-fragment-max-area-mm2", type=float, default=5.0)
    parser.add_argument("--keep-amn-mapping-fragments", action="store_true")
    parser.add_argument("--colorbar-width-fraction", type=float, default=0.62)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    density, vertices = PROFILES[args.surface_space]
    args.surface_density, args.surface_vertices = density, vertices
    args.surface_dir = args.surface_dir or (args.fsaverage5_dir if args.surface_space == "fsaverage5" else args.fsaverage6_dir)
    if any(char in args.output_basename for char in '/\\:*?"<>|') or args.output_basename in ("", ".", ".."):
        parser.error("output_basename must be a plain filename prefix")
    if not np.isclose(args.surface_mix_pial + args.surface_mix_inflated, 1.0):
        parser.error("--surface-mix-pial and --surface-mix-inflated must sum to 1")
    if not 0 < args.colorbar_width_fraction <= 1:
        parser.error("--colorbar-width-fraction must be in (0, 1]")
    if args.amn_fragment_max_vertices < 0 or args.amn_fragment_max_area_mm2 < 0:
        parser.error("AMN fragment thresholds must be non-negative")
    return args


def load_engine():
    if not ENGINE_FILE.is_file():
        raise FileNotFoundError(f"Missing sibling rendering engine: {ENGINE_FILE}")
    spec = importlib.util.spec_from_file_location("amn_scan_engine", ENGINE_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.load_rendering_dependencies()
    return module


def configure_engine(engine, args: argparse.Namespace) -> None:
    engine.apply_style_arguments(SimpleNamespace(
        contour_width=args.contour_width, boundary_offset=args.boundary_offset,
        boundary_inset=args.boundary_inset, amn_scan_contact_mm=args.amn_scan_contact_mm,
        amn_extra_inset=args.amn_extra_inset, amn_collapse_scope=args.amn_collapse_scope,
        surface_mix_pial=args.surface_mix_pial, surface_mix_inflated=args.surface_mix_inflated,
        colorbar_width_fraction=args.colorbar_width_fraction,
    ))


def display_limits(values: dict[str, np.ndarray], cortex: dict[str, np.ndarray], args: argparse.Namespace) -> tuple[float, float, float]:
    finite = np.concatenate([values[hemi][cortex[hemi]] for hemi in ("lh", "rh")])
    if args.mode == "activation":
        maximum = args.display_max if args.display_max is not None else float(np.max(np.abs(finite)))
        if maximum <= 0:
            raise ValueError("Activation display range requires non-zero cortical data or --display-max")
        return (-maximum if args.display_min is None else args.display_min,
                0.0 if args.display_mid is None else args.display_mid, maximum)
    minimum = float(finite.min()) if args.display_min is None else args.display_min
    maximum = float(finite.max()) if args.display_max is None else args.display_max
    midpoint = (minimum + maximum) / 2 if args.display_mid is None else args.display_mid
    if not minimum < midpoint < maximum:
        raise ValueError("FC display limits must satisfy --display-min < --display-mid < --display-max")
    return minimum, midpoint, maximum


def main() -> None:
    args = parse_args()
    engine = load_engine()
    configure_engine(engine, args)
    input_file = args.input_file.expanduser().resolve()
    if not input_file.is_file() or not input_file.name.lower().endswith((".nii", ".nii.gz")):
        raise FileNotFoundError(f"Missing MNI152 NIfTI input: {input_file}")
    image = engine.nib.load(str(input_file))
    if len(image.shape) != 3:
        raise ValueError("Input must be one 3D MNI152 image")
    lut_file = args.lut_file or engine.DEFAULT_LUT_FILES[args.mode]
    required = [lut_file, args.amn_left_label, args.amn_right_label]
    required += [args.surface_dir / "surf" / f"{hemi}.{kind}" for hemi in ("lh", "rh") for kind in ("pial", "inflated")]
    required += [args.surface_dir / "label" / f"{hemi}.cortex.label" for hemi in ("lh", "rh")]
    if args.amn_reference_space == "fsaverage5":
        required += [args.fsaverage5_dir / "surf" / f"{hemi}.pial" for hemi in ("lh", "rh")]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing files:\n" + "\n".join(missing))
    output_dir = args.output_dir.expanduser().resolve()
    png_file = output_dir / f"{args.output_basename}_AMN-SCAN_4panel.png"
    if png_file.exists() and not args.overwrite:
        print(f"Skip existing {png_file}; add --overwrite to redraw")
        return
    cortex = {hemi: engine.load_cortex(args.surface_dir, hemi, args.surface_vertices) for hemi in ("lh", "rh")}
    source = {"lh": engine.load_vector(args.amn_left_label) != 0, "rh": engine.load_vector(args.amn_right_label) != 0}
    mapped = source if args.amn_reference_space == "target" else {
        hemi: engine.fs5_to_target_mask(source[hemi], args.fsaverage5_dir, args.surface_dir, hemi) for hemi in ("lh", "rh")
    }
    for hemi in ("lh", "rh"):
        if len(mapped[hemi]) != args.surface_vertices:
            raise ValueError(f"AMN {hemi} label has {len(mapped[hemi])} vertices; expected {args.surface_vertices}")
    amn, fragment_report = engine.filter_amn_mapping_fragments(
        args.fsaverage5_dir, args.surface_dir, source, mapped, args.amn_reference_space == "target",
        args.amn_fragment_max_vertices, args.amn_fragment_max_area_mm2, args.keep_amn_mapping_fragments,
    )
    scan = {}
    for hemi, side in (("lh", "left"), ("rh", "right")):
        path = engine.resolve_scan_label(args.scan_atlas_dir, getattr(args, f"scan_{side}_label"), hemi, args.surface_space)
        values = engine.load_vector(path)
        if len(values) != args.surface_vertices:
            raise ValueError(f"SCAN {hemi} label has {len(values)} vertices; expected {args.surface_vertices}")
        scan[hemi] = np.isclose(values, args.scan_label_value)
    labels = {hemi: {"AMN": np.flatnonzero(amn[hemi]), "SCAN": np.flatnonzero(scan[hemi])} for hemi in ("lh", "rh")}
    if any(not len(labels[hemi][name]) for hemi in ("lh", "rh") for name in ("AMN", "SCAN")):
        raise ValueError("AMN or SCAN label is empty after validation")
    textures = engine.project_to_surface(input_file, cortex, args.surface_density, args.surface_vertices)
    limits = display_limits(textures, cortex, args)
    cmap = engine.palette(lut_file, args.mode)
    output_dir.mkdir(parents=True, exist_ok=True)
    for hemi in ("lh", "rh"):
        gifti = engine.nib.gifti.GiftiImage(darrays=[engine.nib.gifti.GiftiDataArray(textures[hemi].astype(np.float32))])
        engine.nib.save(gifti, str(output_dir / f"{args.output_basename}_{hemi}.func.gii"))
    engine.save_quad(args.surface_dir, textures, cortex, labels, cmap, limits, png_file,
                     colorbar_label="Z statistic" if args.mode == "activation" else "Z(r)")
    (output_dir / f"{args.output_basename}_display_limits.tsv").write_text(
        "mode\tmin\tmid\tmax\n" + f"{args.mode}\t{limits[0]:.8g}\t{limits[1]:.8g}\t{limits[2]:.8g}\n", encoding="utf-8")
    (output_dir / f"{args.output_basename}_amn_fragment_removal.tsv").write_text(fragment_report, encoding="utf-8")
    (output_dir / f"{args.output_basename}_amn_component_report.tsv").write_text(
        engine.amn_component_report(args.fsaverage5_dir, args.surface_dir, source, amn, args.amn_reference_space == "target"), encoding="utf-8")
    (output_dir / f"{args.output_basename}_amn_contour_adjustment.tsv").write_text(
        engine.contour_adjustment_report(args.surface_dir, labels), encoding="utf-8")
    print(f"Wrote {png_file}")
    print(f"Display limits: {limits[0]:g}, {limits[1]:g}, {limits[2]:g}")


if __name__ == "__main__":
    main()
