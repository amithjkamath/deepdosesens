# -*- coding: utf-8 -*-
"""Axial sweep video comparing the reference dose plan with the predicted dose.

Each frame is one axial slice, shown twice: the planned (reference) dose on the
left and the model's prediction on the right, both as a heat map over the planning
CT, with the target volume and the organs at risk outlined. The sweep runs caudal
to cranial, so the clip walks up through the head.

    python -m deepdosesens.visualization.make_dose_video --case DLDP_081
    python -m deepdosesens.visualization.make_dose_video --case largest-ptv
    python -m deepdosesens.visualization.make_dose_video --case best median worst
    python -m deepdosesens.visualization.make_dose_video --show-config

Dose is shown between --dose-min and --dose-max Gy (default 20-65); below the
lower limit the overlay is transparent so the CT stays readable. The prescription
was 60 Gy in 30 fractions, so 65 Gy is just above the hottest expected region and
20 Gy is around the lowest clinically interesting level.
"""

import argparse
import os
import sys
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from scipy import ndimage

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from deepdosesens.config import (  # noqa: E402
    data_path,
    describe,
    prediction_path,
    require,
    results_path,
)
from deepdosesens.visualization.panels import (  # noqa: E402
    BACKGROUND,
    INK,
    STRUCTURE_FILES,
    crop_box,
    dose_colorbar,
    draw_panel,
    encode,
    load_structures,
    read,
    slice_range,
    structure_legend,
)


def load_case(case_dir, prediction_dir):
    """Everything one case's frames need, read once."""
    case_dir, prediction_dir = require(case_dir), require(prediction_dir)
    data = {
        "ct": read(os.path.join(str(case_dir), "CT.nii.gz")),
        "reference": read(os.path.join(str(case_dir), "Dose.nii.gz")),
        "predicted": read(os.path.join(str(prediction_dir), "Dose.nii.gz")),
        "body": read(os.path.join(str(case_dir), "Dose_Mask.nii.gz")),
        "masks": load_structures(case_dir),
    }
    data["slices"] = slice_range(data["body"])
    data["crop"] = crop_box(data["body"])
    return data


def dose_score(data):
    """Mean absolute error over the body, i.e. the score both papers report."""
    return float(np.abs(data["predicted"] - data["reference"])[data["body"] > 0].mean())


def case_metrics(case, cases, run):
    """What makes a case worth showing, measured rather than judged by eye.

    A case with a small target that sits clear of every organ at risk is easy for
    the model and shows nothing: the dose is one blob and no constraint is in play.
    These four numbers pick cases that are actually informative.
    """
    case_dir = data_path(cases, case)
    data = load_case(case_dir, prediction_path(cases, run, case))
    spacing = sitk.ReadImage(str(case_dir / "CT.nii.gz")).GetSpacing()
    voxel_cc = float(np.prod(spacing)) / 1000.0

    target = read(case_dir / "Target.nii.gz") > 0
    brain = read(case_dir / "Brain.nii.gz") > 0
    body = data["body"] > 0
    reference, predicted = data["reference"], data["predicted"]

    # Organs the plan actually has to work around, rather than merely present.
    implicated = 0
    for organ, files in STRUCTURE_FILES.items():
        if organ == "Target":
            continue
        for name in files:
            path = case_dir / f"{name}.nii.gz"
            if not path.exists():
                continue
            mask = read(path) > 0
            if mask.any() and reference[mask].mean() >= 20.0:
                implicated += 1

    # Streak reproduction: the planned dose carries fine radial structure from the
    # delivery hardware on top of a smooth gradient. Subtracting a blurred copy
    # leaves that structure, and its correlation says whether the model reproduced
    # it or only got the smooth part right.
    high_pass = reference - ndimage.gaussian_filter(reference, 2.0)
    high_pass_predicted = predicted - ndimage.gaussian_filter(predicted, 2.0)
    band = body & (reference > 5.0)
    streaks = float(np.corrcoef(high_pass[band], high_pass_predicted[band])[0, 1])

    return {
        "PTV (cc)": target.sum() * voxel_cc,
        "PTV / brain (%)": 100.0 * target.sum() / brain.sum(),
        "OARs >= 20 Gy": implicated,
        "streak r": streaks,
        "dose score (Gy)": dose_score(data),
    }


def draw_frame(case, data, index, out_path, dose_min, dose_max, subtitle_extra=""):
    fig = plt.figure(figsize=(11.5, 6.0), dpi=110, facecolor=BACKGROUND)
    grid = fig.add_gridspec(
        1, 3, width_ratios=[1.0, 1.0, 0.44], left=0.02, right=0.985,
        top=0.86, bottom=0.06, wspace=0.06,
    )
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    for ax in axes:
        ax.set_facecolor(BACKGROUND)

    rows, cols = data["crop"]
    cut = (index, slice(*rows), slice(*cols))
    masks_slice = {organ: mask[cut] for organ, mask in data["masks"].items()}
    draw_panel(
        axes[0], data["ct"][cut], data["reference"][cut], masks_slice,
        "Reference dose  (treatment plan)", dose_min, dose_max,
    )
    draw_panel(
        axes[1], data["ct"][cut], data["predicted"][cut], masks_slice,
        "Predicted dose  (cascaded 3D U-Net)", dose_min, dose_max,
    )

    # Right-hand column: structure legend above, dose colour bar below.
    side = fig.add_subplot(grid[0, 2])
    side.axis("off")
    structure_legend(side, data["masks"])
    dose_colorbar(fig, [0.845, 0.10, 0.022, 0.34], dose_min, dose_max)

    position = data["slices"].index(index) + 1
    fig.suptitle(f"{case}    reference vs predicted dose", color=INK, fontsize=15, y=0.975)
    fig.text(
        0.5, 0.915,
        f"axial slice {position} of {len(data['slices'])}, sweeping caudal → cranial"
        f"{subtitle_extra}",
        color=INK, fontsize=10.5, ha="center", alpha=0.85,
    )
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def build(case, args):
    data = load_case(
        data_path(args.cases, case), prediction_path(args.cases, args.run, case)
    )
    score = dose_score(data)
    print(f"\n=== {case}: {len(data['slices'])} slices, dose score {score:.3f} Gy")

    frame_dir = os.path.join(args.out_dir, f"frames_{case}")
    os.makedirs(frame_dir, exist_ok=True)
    for old in glob(os.path.join(frame_dir, "*.png")):
        os.remove(old)

    for n, index in enumerate(data["slices"]):
        draw_frame(
            case, data, index, os.path.join(frame_dir, f"frame_{n:04d}.png"),
            args.dose_min, args.dose_max,
            subtitle_extra=f"     ·     dose score {score:.2f} Gy",
        )
    print(f"    rendered {len(data['slices'])} frames")

    os.makedirs(args.out_dir, exist_ok=True)
    out_file = os.path.join(args.out_dir, f"dose_sweep_{case}.mp4")
    encode(frame_dir, out_file, args.fps, crf=args.crf)
    print(
        f"    wrote {out_file}  ({os.path.getsize(out_file) / 1e6:.1f} MB, "
        f"{len(data['slices']) / args.fps:.0f}s)"
    )
    if not args.keep_frames:
        for old in glob(os.path.join(frame_dir, "*.png")):
            os.remove(old)
        os.rmdir(frame_dir)
    return out_file


# selector -> (metric, largest first / None for the median, challenging cases only)
SELECTORS = {
    "best": ("dose score (Gy)", False, False),
    "median": ("dose score (Gy)", None, False),
    "worst": ("dose score (Gy)", True, False),
    "largest-ptv": ("PTV (cc)", True, False),
    "most-oar-interaction": ("OARs >= 20 Gy", True, False),
    # Streak reproduction is only interesting where there is a demanding plan to
    # reproduce: the best streak correlation overall belongs to a small target with
    # no organ at risk in play, where the dose is one smooth blob.
    "best-streaks": ("streak r", True, True),
}


def is_challenging(metrics, case, all_metrics):
    """A target above the cohort's median size that at least one organ has to live with."""
    median_ptv = float(np.median([m["PTV (cc)"] for m in all_metrics.values()]))
    return metrics["PTV (cc)"] >= median_ptv and metrics["OARs >= 20 Gy"] >= 1


def resolve_cases(args):
    """Expand the case selectors, ranking on measured properties of each case."""
    selectors = set(args.case) & (set(SELECTORS) | {"all"})
    if not selectors:
        return args.case

    subjects = ["DLDP_%03d" % i for i in range(81, 101)]
    if args.case == ["all"]:
        return subjects

    metrics = {case: case_metrics(case, args.cases, args.run) for case in subjects}
    columns = list(next(iter(metrics.values())))
    header = "  ".join(f"{c:>16s}" for c in columns)
    print(f"\n{'case':<10}{header}")
    for case in sorted(metrics, key=lambda c: -metrics[c]["PTV (cc)"]):
        cells = "  ".join(f"{metrics[case][c]:16.3f}" for c in columns)
        print(f"{case:<10}{cells}")

    picked = []
    for selector in args.case:
        if selector not in SELECTORS:
            picked.append(selector)
            continue
        metric, descending, challenging_only = SELECTORS[selector]
        pool = [
            case
            for case in metrics
            if not challenging_only or is_challenging(metrics[case], case, metrics)
        ]
        ranked = sorted(pool, key=lambda c: metrics[c][metric], reverse=bool(descending))
        if descending is None:
            chosen = ranked[len(ranked) // 2]
        else:
            # Selectors often agree on the same standout case; take that selector's
            # next choice instead, so N selectors give N different cases to look at.
            remaining = [c for c in ranked if c not in picked]
            chosen = (remaining or ranked)[0]
        note = (
            "   (no OAR above 20 Gy -- an easy case)"
            if metrics[chosen]["OARs >= 20 Gy"] == 0
            else ""
        )
        print(f"\n{selector:>22s} -> {chosen}  ({metric} = {metrics[chosen][metric]:.3f}){note}")
        picked.append(chosen)
    return list(dict.fromkeys(picked))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--case", nargs="*",
        default=["most-oar-interaction", "largest-ptv", "best-streaks"],
        help="case names, and/or the selectors "
             + " / ".join(list(SELECTORS) + ["all"]),
    )
    ap.add_argument("--cases", default="glioblastoma", help="case directory under the data root")
    ap.add_argument("--run", default="run-6", help="prediction run to compare against")
    ap.add_argument("--dose-min", type=float, default=20.0, help="lower limit of the heat map (Gy)")
    ap.add_argument("--dose-max", type=float, default=65.0, help="upper limit of the heat map (Gy)")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--crf", type=int, default=20, help="x264 quality; higher = smaller file")
    ap.add_argument("--out-dir", default=str(results_path("videos")))
    ap.add_argument("--keep-frames", action="store_true")
    ap.add_argument("--show-config", action="store_true", help="print resolved paths and exit")
    args = ap.parse_args()

    if args.show_config:
        print(describe())
        return

    made = [build(case, args) for case in resolve_cases(args)]
    print("\nDone:")
    for path in made:
        print("  " + path)


if __name__ == "__main__":
    main()
