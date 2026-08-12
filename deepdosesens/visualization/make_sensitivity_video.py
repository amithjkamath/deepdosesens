# -*- coding: utf-8 -*-
"""Video of the sensitivity experiment: ten plausible left optic nerve contours.

This is the claim both papers rest on -- that the dose predictor reacts to a small
contour edit the way the treatment planning system does, so a contour can be
judged by its dosimetric consequence rather than by geometry alone.

One frame per contour variant. Left panel: the reference plan, re-optimised for that
contour. Right panel: the model's prediction for the same contour. The variant's own
optic nerve outline is drawn solid and the original contour dashed, so the edit is
visible, and the mean dose each panel delivers to the contour is printed beneath it.
The view is zoomed on the optic nerve region, which is only a few voxels across at
this resolution.

    python -m deepdosesens.visualization.make_sensitivity_video
    python -m deepdosesens.visualization.make_sensitivity_video --seconds-per-frame 1.5
"""

import argparse
import os
import sys
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from deepdosesens.analyze.scores import volumetric_dice  # noqa: E402
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
    dose_colorbar,
    draw_panel,
    encode,
    load_structures,
    read,
    structure_legend,
)

# Same hue for both, since they are the same structure in two states: solid is the
# contour under test, dashed is the original it was edited from. A second hue would
# read as a second organ, and white would collide with the target volume.
VARIANT_COLOUR = "#00e5ff"
ORIGINAL_COLOUR = "#00e5ff"

def load_variants(cases_dir, prediction_dir, n_variants=10):
    """Read all ten contour variants with their reference and predicted doses."""
    variants = []
    for index in range(n_variants):
        case = "DLDP_%03d" % index
        case_dir = require(os.path.join(str(cases_dir), case))
        variants.append(
            {
                "case": case,
                "ct": read(os.path.join(case_dir, "CT.nii.gz")),
                "reference": read(os.path.join(case_dir, "Dose.nii.gz")),
                "predicted": read(
                    os.path.join(str(prediction_dir), case, "Dose.nii.gz")
                ),
                "nerve": read(os.path.join(case_dir, "OpticNerve_L.nii.gz")),
                "masks": load_structures(case_dir),
            }
        )
    return variants


def focus(nerves, margin=18):
    """The slice and in-plane window that show every variant of the nerve.

    The nerve is a handful of voxels across at 128^3, so the frame is cropped
    around it. The slice has to work for all ten contours at once -- picking the
    best slice for the original alone leaves variants that sit slightly higher or
    lower invisible -- so it is chosen among the slices where every variant has
    voxels, taking the one with the most in total.
    """
    areas = np.array([n.reshape(n.shape[0], -1).sum(axis=1) for n in nerves])
    shared = np.flatnonzero((areas > 0).all(axis=0))
    if shared.size:
        index = int(shared[np.argmax(areas[:, shared].sum(axis=0))])
    else:  # no slice holds all of them; fall back to the largest total
        index = int(np.argmax(areas.sum(axis=0)))
        print("    note: no slice contains every variant; some frames omit the contour")

    union = np.maximum.reduce([n[index] for n in nerves])
    rows = np.flatnonzero(union.any(axis=1))
    cols = np.flatnonzero(union.any(axis=0))
    centre_row, centre_col = (rows.min() + rows.max()) // 2, (cols.min() + cols.max()) // 2
    half = margin
    return index, (
        (max(0, centre_row - half), min(nerves[0].shape[1], centre_row + half)),
        (max(0, centre_col - half), min(nerves[0].shape[2], centre_col + half)),
    )


def draw_frame(variant, original_nerve, view, out_path, args, stats):
    index, (rows, cols) = view
    cut = (index, slice(*rows), slice(*cols))

    fig = plt.figure(figsize=(11.5, 6.2), dpi=110, facecolor=BACKGROUND)
    grid = fig.add_gridspec(
        1, 3, width_ratios=[1.0, 1.0, 0.52], left=0.02, right=0.985,
        top=0.85, bottom=0.10, wspace=0.06,
    )
    axes = [fig.add_subplot(grid[0, i]) for i in range(2)]
    for ax in axes:
        ax.set_facecolor(BACKGROUND)

    # The optic nerve group is dropped from the organ outlines: the left nerve is
    # the subject of this experiment and is drawn explicitly below, and leaving the
    # group in would draw it twice in two different colours.
    masks_slice = {
        organ: mask[cut]
        for organ, mask in variant["masks"].items()
        if organ != "OpticNerve"
    }
    # The nerve under test is drawn on top of the organ outlines, with the original
    # contour dashed beside it so the edit itself is what changes between frames.
    extra = [
        (original_nerve[cut], ORIGINAL_COLOUR, 1.9, "dashed"),
        (variant["nerve"][cut], VARIANT_COLOUR, 2.4, "solid"),
    ]
    for ax, key, title in [
        (axes[0], "reference", "Reference dose  (re-optimised plan)"),
        (axes[1], "predicted", "Predicted dose  (cascaded 3D U-Net)"),
    ]:
        draw_panel(
            ax, variant["ct"][cut], variant[key][cut], masks_slice, title,
            args.dose_min, args.dose_max, extra_contours=extra,
        )
        ax.set_xlabel(
            f"mean dose to this contour   {stats[key]:.1f} Gy",
            color=INK, fontsize=11, labelpad=8,
        )

    side = fig.add_subplot(grid[0, 2])
    side.axis("off")
    # The slice is fixed for the whole clip, so listing only the organs actually in
    # view keeps the legend honest and short enough to leave room for the two scales.
    visible = {organ: mask for organ, mask in masks_slice.items() if mask.max() > 0}
    structure_legend(
        side, visible, fontsize=8.5,
        extra=[
            Line2D([0], [0], color=VARIANT_COLOUR, linewidth=2.6,
                   label="Optic nerve L, this variant"),
            Line2D([0], [0], color=ORIGINAL_COLOUR, linewidth=1.9, linestyle="dashed",
                   label="Optic nerve L, original"),
        ],
    )
    column = side.get_position()
    dose_colorbar(fig, [column.x0 + 0.008, 0.14, 0.016, 0.30], args.dose_min, args.dose_max)

    label = "original contour" if stats["variant"] == 0 else f"alternative {stats['variant']}"
    fig.suptitle(
        "Sensitivity to left optic nerve contour variation", color=INK, fontsize=15, y=0.975
    )
    fig.text(
        0.5, 0.905,
        f"{label}     ·     DSC to original {stats['dsc']:.2f}"
        f"     ·     mean dose shift: plan {stats['reference_shift']:+.2f} Gy,"
        f" prediction {stats['predicted_shift']:+.2f} Gy",
        color=INK, fontsize=10.5, ha="center", alpha=0.9,
    )
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def build(args):
    variants = load_variants(
        data_path(args.cases), prediction_path(args.cases, args.run)
    )
    original = variants[0]
    view = focus([v["nerve"] for v in variants], margin=args.zoom)
    print(f"=== {len(variants)} contour variants, slice {view[0]}")

    stats = []
    for index, variant in enumerate(variants):
        inside = variant["nerve"] > 0
        stats.append(
            {
                "variant": index,
                "reference": float(variant["reference"][inside].mean()),
                "predicted": float(variant["predicted"][inside].mean()),
                "dsc": volumetric_dice(original["nerve"], variant["nerve"]),
            }
        )
    for row in stats:
        row["reference_shift"] = row["reference"] - stats[0]["reference"]
        row["predicted_shift"] = row["predicted"] - stats[0]["predicted"]
        print(
            f"    variant {row['variant']}: plan {row['reference']:5.1f} Gy"
            f"   prediction {row['predicted']:5.1f} Gy   DSC {row['dsc']:.2f}"

        )

    frame_dir = os.path.join(args.out_dir, "frames_sensitivity")
    os.makedirs(frame_dir, exist_ok=True)
    for old in glob(os.path.join(frame_dir, "*.png")):
        os.remove(old)

    # Each variant is held for several frames so it can be read, and the sweep
    # walks back down again so the clip loops without a jump.
    order = list(range(len(variants))) + list(range(len(variants) - 2, 0, -1))
    frames_per_variant = max(1, int(round(args.seconds_per_frame * args.fps)))
    n = 0
    for index in order:
        for _ in range(frames_per_variant):
            draw_frame(
                variants[index], original["nerve"], view,
                os.path.join(frame_dir, f"frame_{n:04d}.png"), args, stats[index],
            )
            n += 1
    print(f"    rendered {n} frames")

    os.makedirs(args.out_dir, exist_ok=True)
    out_file = os.path.join(args.out_dir, "optic_nerve_sensitivity.mp4")
    encode(frame_dir, out_file, args.fps, crf=args.crf)
    print(
        f"    wrote {out_file}  ({os.path.getsize(out_file) / 1e6:.1f} MB, "
        f"{n / args.fps:.0f}s)"
    )
    if not args.keep_frames:
        for old in glob(os.path.join(frame_dir, "*.png")):
            os.remove(old)
        os.rmdir(frame_dir)
    return out_file


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", default="optic-nerve-variants")
    ap.add_argument("--run", default="run-6", help="prediction run to show")
    ap.add_argument("--dose-min", type=float, default=20.0)
    ap.add_argument("--dose-max", type=float, default=65.0)
    ap.add_argument("--zoom", type=int, default=22, help="half-width of the crop, in voxels")
    ap.add_argument("--seconds-per-frame", type=float, default=1.2, help="hold per variant")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--out-dir", default=str(results_path("videos")))
    ap.add_argument("--keep-frames", action="store_true")
    ap.add_argument("--show-config", action="store_true")
    args = ap.parse_args()

    if args.show_config:
        print(describe())
        return
    print("\nDone:\n  " + build(args))


if __name__ == "__main__":
    main()
