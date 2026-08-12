# -*- coding: utf-8 -*-
"""Dose and DVH scores, as both papers report them.

The two openKBP metrics [Babier et al., Med Phys 2021] are:

**Dose score** -- mean absolute error between predicted and reference dose inside
a mask. Reported over the body region that can receive dose (``Dose_Mask``); the
per-OAR columns of the ISBI table use each organ's own mask.

**DVH score** -- mean absolute difference of dose-volume metrics. For an OAR
these are D(0.1 cc) and the mean dose; for the target volume, D1, D95 and D99.

Two aggregation conventions matter when comparing with the printed tables, and
both are made explicit here rather than left to a default:

* ``std(ddof=1)`` -- the sample standard deviation, which is what the papers'
  parenthesised values are (pandas' default; NumPy's default differs).
* the subject-level DVH score is the mean over *every individual metric*, not the
  mean of the per-structure means. With 13 OARs contributing 2 metrics each and
  the target contributing 3, the two differ.
"""

import os

import numpy as np
import pandas as pd
import SimpleITK as sitk

from deepdosesens.data.utils import get_spacing, read_nifti_image
from deepdosesens.training.metrics import compute_abs_dose_difference, compute_dvh

# The 13 OARs of ISBI Table 1, in the order the paper lists them.
OARS = [
    "BrainStem",
    "Chiasm",
    "Cochlea_L",
    "Cochlea_R",
    "Eye_L",
    "Eye_R",
    "Hippocampus_L",
    "Hippocampus_R",
    "LacrimalGland_L",
    "LacrimalGland_R",
    "OpticNerve_L",
    "OpticNerve_R",
    "Pituitary",
]
# Dose-score-only structures: Brain has no DVH criteria in either paper.
EXTRA_STRUCTURES = ["Brain", "Target"]

TEST_SUBJECTS = ["DLDP_%03d" % i for i in range(81, 101)]


def case_scores(reference_dir, prediction_dir):
    """Dose and DVH scores for one case. Returns ``(dose, dvh)`` dicts."""
    reference_dir, prediction_dir = str(reference_dir), str(prediction_dir)
    reference = read_nifti_image(os.path.join(reference_dir, "Dose.nii.gz"))
    predicted = read_nifti_image(os.path.join(prediction_dir, "Dose.nii.gz"))
    body = read_nifti_image(os.path.join(reference_dir, "Dose_Mask.nii.gz"))

    dose = {"Body": compute_abs_dose_difference(predicted, reference, body)}
    metrics = {}  # structure -> list of |reference - predicted| metric differences

    for name in OARS + EXTRA_STRUCTURES:
        mask_file = os.path.join(reference_dir, name + ".nii.gz")
        if not os.path.exists(mask_file):
            continue
        mask = read_nifti_image(mask_file, type=sitk.sitkUInt8)
        dose[name] = compute_abs_dose_difference(predicted, reference, mask)

        if name == "Brain":
            continue
        mode = "target" if name == "Target" else "OAR"
        spacing = get_spacing(mask_file)
        predicted_dvh = compute_dvh(predicted, mask, mode_=mode, spacing=spacing)
        reference_dvh = compute_dvh(reference, mask, mode_=mode, spacing=spacing)
        metrics[name] = [
            abs(reference_dvh[m] - predicted_dvh[m]) for m in reference_dvh
        ]

    dvh = {name: float(np.mean(values)) for name, values in metrics.items()}
    oar_metrics = [v for name, values in metrics.items() if name != "Target" for v in values]
    dvh["OARs"] = float(np.mean(oar_metrics))
    dvh["Overall"] = float(np.mean([v for values in metrics.values() for v in values]))
    return dose, dvh


def run_scores(reference_root, prediction_root, subjects=None):
    """Score a whole prediction run. Returns ``(dose_df, dvh_df)``, cases as rows."""
    subjects = subjects if subjects is not None else TEST_SUBJECTS
    dose_rows, dvh_rows = {}, {}
    for subject in subjects:
        dose, dvh = case_scores(
            os.path.join(str(reference_root), subject),
            os.path.join(str(prediction_root), subject),
        )
        dose_rows[subject], dvh_rows[subject] = dose, dvh
    return pd.DataFrame(dose_rows).T, pd.DataFrame(dvh_rows).T


def mean_sd(values, decimals=3):
    """``mean (sd)`` with the sample standard deviation the papers report."""
    values = np.asarray(values, dtype=float)
    return f"%.{decimals}f (%.{decimals}f)" % (values.mean(), values.std(ddof=1))


def volumetric_dice(first, second):
    """Dice similarity coefficient between two binary masks."""
    first, second = first > 0, second > 0
    return 2.0 * np.sum(first & second) / (np.sum(first) + np.sum(second))


def optic_nerve_variants(reference_root, prediction_root, n_variants=10):
    """Per-contour mean dose to the left optic nerve, reference and predicted.

    Index 0 is the reference contour; 1..n are the plausible alternatives, each
    with its own re-optimised reference plan. Returns a DataFrame indexed by
    variant, with the mean dose the plan delivers, the mean dose the model
    predicts, and the DSC against the reference contour.
    """
    rows = {}
    reference_mask = None
    for index in range(n_variants):
        case = "DLDP_%03d" % index
        case_dir = os.path.join(str(reference_root), case)
        mask = read_nifti_image(os.path.join(case_dir, "OpticNerve_L.nii.gz"))
        reference_dose = read_nifti_image(os.path.join(case_dir, "Dose.nii.gz"))
        predicted_dose = read_nifti_image(
            os.path.join(str(prediction_root), case, "Dose.nii.gz")
        )
        if reference_mask is None:
            reference_mask = mask
        rows[index] = {
            "reference mean dose (Gy)": float(np.mean(reference_dose[mask > 0])),
            "predicted mean dose (Gy)": float(np.mean(predicted_dose[mask > 0])),
            "DSC": volumetric_dice(reference_mask, mask),
        }
    table = pd.DataFrame(rows).T
    table.index.name = "variant"
    table["|R_i - R_0|"] = (
        table["reference mean dose (Gy)"] - table.loc[0, "reference mean dose (Gy)"]
    ).abs()
    table["|P_i - P_0|"] = (
        table["predicted mean dose (Gy)"] - table.loc[0, "predicted mean dose (Gy)"]
    ).abs()
    return table
