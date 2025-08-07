# -*- encoding: utf-8 -*-
"""
This module contains functions to compute dose and DVH scores for evaluation.
It includes functions to compute absolute dose differences and DVH metrics for
specific structures, as well as a function to compute scores for a given prediction
and ground truth directory.

These functions are inspired by https://github.com/ababier/open-kbp
"""
import os
import numpy as np
import SimpleITK as sitk
from deepdosesens.data.utils import get_spacing, read_nifti_image


def compute_abs_dose_difference(
    pred_: np.ndarray, gt_: np.ndarray, dose_mask_=None
) -> float:
    """
    Compute absolute dose difference between prediction and ground truth.
        If dose_mask_ is provided, only compute the difference in the masked region.
    """
    if dose_mask_ is not None:
        pred_ = pred_[dose_mask_ > 0]
        gt_ = gt_[dose_mask_ > 0]

    dose_difference = np.mean(np.abs(pred_ - gt_))
    return float(dose_difference)


def compute_dvh(dose_: np.ndarray, mask_: np.ndarray, mode_: str, spacing=None):
    """
    Compute DVH metrics for the given dose and mask.
        mode_ can be "Target" or "OAR".
        If mode_ is "Target", it computes D1, D95, and D99.
        If mode_ is "OAR", it computes D_0.1_cc and Dmean.
        spacing is required for OAR metrics to calculate D_0.1_cc.
    """
    output = {}

    if mode_.lower() == "target":
        roi_dose_ = dose_[mask_ > 0]
        output["D1"] = np.percentile(roi_dose_, 99)
        output["D95"] = np.percentile(roi_dose_, 5)
        output["D99"] = np.percentile(roi_dose_, 1)

    elif mode_.lower() == "oar":
        if spacing is None:
            raise ValueError("Calculating OAR metrics need spacing.")

        roi_dose_ = dose_[mask_ > 0]
        roi_size_ = len(roi_dose_)
        voxel_size_ = np.prod(spacing)
        voxels_in_tenth_of_cc = np.maximum(1, np.round(100 / voxel_size_))
        # D_0.1_cc
        fractional_volume_ = 100 - voxels_in_tenth_of_cc / roi_size_ * 100
        if fractional_volume_ <= 0:
            output["D_0.1_cc"] = 0.0
        else:
            output["D_0.1_cc"] = np.percentile(roi_dose_, fractional_volume_)

        output["mean"] = np.mean(roi_dose_)
    else:
        raise ValueError("Unknown mode. Can only be 'Target' or 'OAR'!")

    return output


def compute_scores(prediction_path, reference_path, structure_list=None):

    dose_dict = {}
    dvh_dict = {}

    predicted_dose = read_nifti_image(os.path.join(prediction_path, "Dose.nii.gz"))
    reference_dose = read_nifti_image(os.path.join(reference_path, "Dose.nii.gz"))
    dose_mask = read_nifti_image(os.path.join(reference_path, "Dose_Mask.nii.gz"))

    dose_dict["Body"] = compute_abs_dose_difference(
        predicted_dose, reference_dose, dose_mask
    )

    for structure_name in structure_list:

        dose_mask = read_nifti_image(
            os.path.join(reference_path, structure_name + ".nii.gz")
        )
        dose_diff = compute_abs_dose_difference(
            predicted_dose, reference_dose, dose_mask
        )
        dose_dict[structure_name] = dose_diff

        structure_file = os.path.join(reference_path, structure_name + ".nii.gz")
        if os.path.exists(structure_file):
            structure = read_nifti_image(structure_file, type=sitk.sitkUInt8)
            spacing = get_spacing(structure_file)
            if structure_name.find("Target") > -1:
                mode = "target"
            else:
                mode = "OAR"
            dvh_pred = compute_dvh(
                predicted_dose, structure, mode_=mode, spacing=spacing
            )
            dvh_ref = compute_dvh(
                reference_dose, structure, mode_=mode, spacing=spacing
            )

            dvh_diff = 0
            metrics = list(dvh_ref.keys())
            for metric in metrics:
                this_diff = abs(dvh_ref[metric] - dvh_pred[metric])
                dvh_diff += this_diff
            dvh_dict[structure_name] = dvh_diff / len(metrics)

    return dose_dict, dvh_dict
