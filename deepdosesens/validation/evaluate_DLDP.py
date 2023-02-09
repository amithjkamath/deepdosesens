import numpy as np
import os
import SimpleITK as sitk
from tqdm import tqdm

"""
These codes are modified from https://github.com/ababier/open-kbp
"""


def get_3D_Dose_dif(pred, gt, possible_dose_mask=None):
    if possible_dose_mask is not None:
        pred = pred[possible_dose_mask > 0]
        gt = gt[possible_dose_mask > 0]

    dif = np.mean(np.abs(pred - gt))
    return dif


def get_DVH_metrics(_dose, _mask, mode, spacing=None):
    output = {}

    if mode == "target":
        _roi_dose = _dose[_mask > 0]
        # D1
        output["D1"] = np.percentile(_roi_dose, 99)
        # D95
        output["D95"] = np.percentile(_roi_dose, 5)
        # D99
        output["D99"] = np.percentile(_roi_dose, 1)

    elif mode == "OAR":
        if spacing is None:
            raise Exception("calculate OAR metrics need spacing")

        _roi_dose = _dose[_mask > 0]
        _roi_size = len(_roi_dose)
        _voxel_size = np.prod(spacing)
        voxels_in_tenth_of_cc = np.maximum(1, np.round(100 / _voxel_size))
        # D_0.1_cc
        fractional_volume_to_evaluate = 100 - voxels_in_tenth_of_cc / _roi_size * 100
        if fractional_volume_to_evaluate <= 0:
            output["D_0.1_cc"] = 0.0
        else:
            output["D_0.1_cc"] = np.percentile(_roi_dose, fractional_volume_to_evaluate)
        # Dmean
        output["mean"] = np.mean(_roi_dose)
    else:
        raise Exception("Unknown mode!")

    return output


def get_Dose_score_and_DVH_score(prediction_dir, patient_id, gt_dir):

    list_dose_dif = []
    list_DVH_dif = []

    pred_nii = sitk.ReadImage(
        os.path.join(prediction_dir, "DLDP_" + str(patient_id).zfill(3), "Dose.nii.gz")
    )
    pred = sitk.GetArrayFromImage(pred_nii)

    gt_nii = sitk.ReadImage(
        os.path.join(gt_dir, "DLDP_" + str(patient_id).zfill(3), "Dose.nii.gz")
    )
    gt = sitk.GetArrayFromImage(gt_nii)

    # Dose dif
    possible_dose_mask_nii = sitk.ReadImage(
        os.path.join(gt_dir, "DLDP_" + str(patient_id).zfill(3), "Dose_Mask.nii.gz")
    )
    possible_dose_mask = sitk.GetArrayFromImage(possible_dose_mask_nii)
    list_dose_dif.append(get_3D_Dose_dif(pred, gt, possible_dose_mask))

    dvh_dict = {}

    # DVH dif
    for structure_name in [
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
        "Target",
    ]:
        structure_file = os.path.join(
            gt_dir, "DLDP_" + str(patient_id).zfill(3), structure_name + ".nii.gz"
        )

        # If the structure has been delineated
        if os.path.exists(structure_file):
            structure_nii = sitk.ReadImage(structure_file, sitk.sitkUInt8)
            structure = sitk.GetArrayFromImage(structure_nii)

            spacing = structure_nii.GetSpacing()
            if structure_name.find("Target") > -1:
                mode = "target"
            else:
                mode = "OAR"
            pred_DVH = get_DVH_metrics(pred, structure, mode=mode, spacing=spacing)
            gt_DVH = get_DVH_metrics(gt, structure, mode=mode, spacing=spacing)

            dvh_dict[structure_name] = [pred_DVH, gt_DVH]

            for metric in gt_DVH.keys():
                list_DVH_dif.append(abs(gt_DVH[metric] - pred_DVH[metric]))

    return np.mean(list_dose_dif), np.mean(list_DVH_dif)


def get_Dose_score_and_DVH_score_per_ROI(prediction_dir, patient_id, gt_dir):

    list_dose_dif = []
    list_DVH_dif = []
    overall_dose_dict = {}
    overall_dvh_dict = {}

    structure_list = [
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
        "Target",
    ]

    pred_nii = sitk.ReadImage(
        os.path.join(prediction_dir, "DLDP_" + str(patient_id).zfill(3), "Dose.nii.gz")
    )
    pred = sitk.GetArrayFromImage(pred_nii)

    gt_nii = sitk.ReadImage(
        os.path.join(gt_dir, "DLDP_" + str(patient_id).zfill(3), "Dose.nii.gz")
    )
    gt = sitk.GetArrayFromImage(gt_nii)

    # Dose dif
    possible_dose_mask_nii = sitk.ReadImage(
        os.path.join(gt_dir, "DLDP_" + str(patient_id).zfill(3), "Dose_Mask.nii.gz")
    )
    possible_dose_mask = sitk.GetArrayFromImage(possible_dose_mask_nii)
    list_dose_dif.append(get_3D_Dose_dif(pred, gt, possible_dose_mask))

    dose_dict = {}
    for structure_name in structure_list:
        possible_dose_mask_nii = sitk.ReadImage(
            os.path.join(
                gt_dir, "DLDP_" + str(patient_id).zfill(3), structure_name + ".nii.gz"
            )
        )
        possible_dose_mask = sitk.GetArrayFromImage(possible_dose_mask_nii)
        dose_diff = get_3D_Dose_dif(pred, gt, possible_dose_mask)
        list_dose_dif.append(dose_diff)
        dose_dict[structure_name] = dose_diff

    overall_dose_dict["DLDP_" + str(patient_id).zfill(3)] = dose_dict

    dvh_dict = {}
    for structure_name in structure_list:
        structure_file = os.path.join(
            gt_dir, "DLDP_" + str(patient_id).zfill(3), structure_name + ".nii.gz"
        )

        # If the structure has been delineated
        if os.path.exists(structure_file):
            structure_nii = sitk.ReadImage(structure_file, sitk.sitkUInt8)
            structure = sitk.GetArrayFromImage(structure_nii)

            spacing = structure_nii.GetSpacing()
            if structure_name.find("Target") > -1:
                mode = "target"
            else:
                mode = "OAR"
            pred_DVH = get_DVH_metrics(pred, structure, mode=mode, spacing=spacing)
            gt_DVH = get_DVH_metrics(gt, structure, mode=mode, spacing=spacing)

            dvh_diff = 0
            for metric in gt_DVH.keys():
                this_diff = abs(gt_DVH[metric] - pred_DVH[metric])
                list_DVH_dif.append(this_diff)
                dvh_diff += this_diff
            dvh_dict[structure_name] = dvh_diff / len(gt_DVH.keys())

    overall_dvh_dict["DLDP_" + str(patient_id).zfill(3)] = dvh_dict

    return overall_dose_dict, overall_dvh_dict
