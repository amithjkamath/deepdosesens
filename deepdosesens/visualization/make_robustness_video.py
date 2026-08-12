# -*- coding: utf-8 -*-
"""Axial sweep videos for the Cancers 2023 robustness claim.

The Cancers paper reports that the initial dose predictor loses conformity where a
target is concave or splits into several lesions -- shapes the 60 standard training
cases do not cover -- and that adding six such cases to the training set recovers
it. These clips show that directly: three panels per frame, the planned dose and
then the same slice predicted by the initial model and by the retrained one, so the
column that changes is the model.

    python -m deepdosesens.visualization.make_robustness_video
    python -m deepdosesens.visualization.make_robustness_video --scenario concave

The default cases are chosen by target geometry rather than by name: the most
concave single-lesion target and the target with the most separate lesions among
the out-of-distribution cases.
"""

import argparse
import os
import sys
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError

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
    crop_box,
    dose_colorbar,
    draw_panel,
    encode,
    load_structures,
    read,
    slice_range,
    structure_legend,
)

# The out-of-distribution cases the Cancers paper added: 101-110 carry concave
# single-lesion targets, 111-120 targets split into several lesions.
CONCAVE_CASES = ["DLDP_%03d" % i for i in range(101, 111)]
MULTIPLE_CASES = ["DLDP_%03d" % i for i in range(111, 121)]

SCENARIOS = {
    "concave": {
        "title": "Concave target: initial model vs model retrained with concave cases",
        "pool": CONCAVE_CASES,
        "updated": "concave-updated-model",
        "updated_label": "Concave-updated model",
        "pick": "most concave",
    },
    "multiple": {
        "title": "Multiple lesions: initial model vs model retrained with multi-lesion cases",
        "pool": MULTIPLE_CASES,
        "updated": "multiple-lesion-updated-model",
        "updated_label": "Multiple-lesion-updated model",
        "pick": "most lesions",
    },
}


def target_shape(case):
    """Lesion count and solidity of a case's target volume."""
    target = read(data_path("glioblastoma", case, "Target.nii.gz"))
    labelled, _ = ndimage.label(target > 0)
    sizes = np.bincount(labelled.ravel())[1:]
    # Ignore specks: a lesion is a component at least 5% of the largest one.
    lesions = int((sizes > 0.05 * sizes.max()).sum()) if sizes.size else 0
    points = np.argwhere(target > 0)
    try:
        solidity = points.shape[0] / ConvexHull(points).volume
    except (QhullError, ValueError):
        solidity = float("nan")
    return lesions, solidity


# Six of the ten cases in each pool were added to the retrained model's training
# set, and the archive does not say which. The scores give it away: on a case it
# trained on, a retrained model's dose score collapses to a fraction of the initial
# model's, far beyond any genuine generalisation. Cases below this ratio are
# excluded, so the videos only ever show held-out behaviour.
MEMORISED_RATIO = 0.5


def dose_score(case, model):
    reference = read(data_path("glioblastoma", case, "Dose.nii.gz"))
    body = read(data_path("glioblastoma", case, "Dose_Mask.nii.gz")) > 0
    prediction = read(prediction_path("glioblastoma", model, case, "Dose.nii.gz"))
    return float(np.abs(prediction - reference)[body].mean())


def has_mask_holes(case):
    """Whether the archived dose mask has empty slices inside the dose region.

    Inference zeroes the prediction wherever the mask is empty, and the dose score
    ignores those voxels, so such a case scores normally but renders as black bands
    across the middle of the dose distribution. One archived case (DLDP_118) is
    affected; it would read as a rendering fault rather than as model behaviour.
    """
    body = read(data_path("glioblastoma", case, "Dose_Mask.nii.gz"))
    reference = read(data_path("glioblastoma", case, "Dose.nii.gz"))
    empty = (body > 0).sum(axis=(1, 2)) == 0
    return bool((empty & (reference.max(axis=(1, 2)) > 20)).any())


def pick_case(scenario):
    """The case that shows the scenario most clearly, among held-out cases."""
    usable = [
        case
        for case in scenario["pool"]
        if data_path("glioblastoma", case).is_dir()
        and prediction_path("glioblastoma", scenario["updated"], case).is_dir()
    ]
    shapes, held_out = {}, []
    for case in usable:
        shapes[case] = target_shape(case)
        initial = dose_score(case, "initial-model")
        updated = dose_score(case, scenario["updated"])
        ratio = updated / initial
        skip = None
        if ratio < MEMORISED_RATIO:
            skip = "used in retraining"
        elif has_mask_holes(case):
            skip = "archived dose mask has empty slices"
        print(
            f"    {case}  lesions {shapes[case][0]}  solidity {shapes[case][1]:.2f}"
            f"  dose score {initial:.2f} -> {updated:.2f}"
            + (f"   ({skip}; skipped)" if skip else "")
        )
        if skip is None:
            held_out.append(case)
    if not held_out:
        raise RuntimeError(f"every {scenario['pick']} case looks like a training case")
    if scenario["pick"] == "most concave":
        ranked = sorted(held_out, key=lambda c: shapes[c][1])
    else:
        ranked = sorted(held_out, key=lambda c: (-shapes[c][0], shapes[c][1]))
    return ranked[0]


def load_case(case, models):
    case_dir = require(data_path("glioblastoma", case))
    data = {
        "ct": read(case_dir / "CT.nii.gz"),
        "body": read(case_dir / "Dose_Mask.nii.gz"),
        "reference": read(case_dir / "Dose.nii.gz"),
        "masks": load_structures(case_dir),
        "predictions": {
            model: read(
                require(prediction_path("glioblastoma", model, case)) / "Dose.nii.gz"
            )
            for model in models
        },
    }
    data["slices"] = slice_range(data["body"])
    data["crop"] = crop_box(data["body"])
    inside = data["body"] > 0
    data["scores"] = {
        model: float(np.abs(prediction - data["reference"])[inside].mean())
        for model, prediction in data["predictions"].items()
    }
    return data


def draw_frame(case, data, index, panels, out_path, args, title):
    fig = plt.figure(figsize=(14.0, 5.6), dpi=110, facecolor=BACKGROUND)
    grid = fig.add_gridspec(
        1, 4, width_ratios=[1.0, 1.0, 1.0, 0.42], left=0.015, right=0.985,
        top=0.83, bottom=0.09, wspace=0.05,
    )
    rows, cols = data["crop"]
    cut = (index, slice(*rows), slice(*cols))
    masks_slice = {organ: mask[cut] for organ, mask in data["masks"].items()}

    for column, (dose, label, score) in enumerate(panels):
        ax = fig.add_subplot(grid[0, column])
        ax.set_facecolor(BACKGROUND)
        draw_panel(
            ax, data["ct"][cut], dose[cut], masks_slice, label,
            args.dose_min, args.dose_max,
        )
        if score is not None:
            ax.set_xlabel(f"dose score {score:.2f} Gy", color=INK, fontsize=11, labelpad=7)

    side = fig.add_subplot(grid[0, 3])
    side.axis("off")
    structure_legend(side, data["masks"], fontsize=9)
    dose_colorbar(fig, [0.885, 0.12, 0.017, 0.32], args.dose_min, args.dose_max)

    position = data["slices"].index(index) + 1
    fig.suptitle(title, color=INK, fontsize=14, y=0.965)
    fig.text(
        0.5, 0.895,
        f"{case}     ·     axial slice {position} of {len(data['slices'])},"
        " sweeping caudal → cranial",
        color=INK, fontsize=10.5, ha="center", alpha=0.85,
    )
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def build(name, args):
    scenario = SCENARIOS[name]
    print(f"\n=== {name}: choosing the case with the {scenario['pick']} target")
    case = args.case or pick_case(scenario)
    models = ["initial-model", scenario["updated"]]
    data = load_case(case, models)
    print(
        f"    {case}: {len(data['slices'])} slices, dose score"
        f" initial {data['scores']['initial-model']:.3f} Gy,"
        f" updated {data['scores'][scenario['updated']]:.3f} Gy"
    )

    panels = [
        (data["reference"], "Reference dose  (treatment plan)", None),
        (data["predictions"]["initial-model"], "Initial model", data["scores"]["initial-model"]),
        (
            data["predictions"][scenario["updated"]],
            scenario["updated_label"],
            data["scores"][scenario["updated"]],
        ),
    ]

    frame_dir = os.path.join(args.out_dir, f"frames_{name}")
    os.makedirs(frame_dir, exist_ok=True)
    for old in glob(os.path.join(frame_dir, "*.png")):
        os.remove(old)

    for n, index in enumerate(data["slices"]):
        draw_frame(
            case, data, index, panels,
            os.path.join(frame_dir, f"frame_{n:04d}.png"), args, scenario["title"],
        )
    print(f"    rendered {len(data['slices'])} frames")

    out_file = os.path.join(args.out_dir, f"robustness_{name}.mp4")
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", default="all", choices=list(SCENARIOS) + ["all"])
    ap.add_argument("--case", default=None, help="override the automatic case choice")
    ap.add_argument("--dose-min", type=float, default=20.0)
    ap.add_argument("--dose-max", type=float, default=65.0)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--out-dir", default=str(results_path("videos")))
    ap.add_argument("--keep-frames", action="store_true")
    ap.add_argument("--show-config", action="store_true")
    args = ap.parse_args()

    if args.show_config:
        print(describe())
        return

    os.makedirs(args.out_dir, exist_ok=True)
    targets = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    made = [build(name, args) for name in targets]
    print("\nDone:")
    for path in made:
        print("  " + path)


if __name__ == "__main__":
    main()
