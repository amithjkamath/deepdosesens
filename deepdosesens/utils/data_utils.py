import os
import numpy as np
import SimpleITK as sitk
import torch


def read_data(patient_dir):
    dict_images = {}
    list_structures = [
        "CT",
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
        "Dose",
        "Dose_Mask",
    ]

    for structure_name in list_structures:
        structure_file = patient_dir + "/" + structure_name + ".nii.gz"

        if structure_name == "CT":
            dtype = sitk.sitkInt16
        else:
            dtype = sitk.sitkUInt8

        if os.path.exists(structure_file):
            dict_images[structure_name] = sitk.ReadImage(structure_file, dtype)
            dict_images[structure_name] = sitk.GetArrayFromImage(
                dict_images[structure_name]
            )[np.newaxis, :, :, :]
        else:
            dict_images[structure_name] = np.zeros((1, 128, 128, 128), np.uint8)

    return dict_images


def pre_processing(dict_images):
    # PTVs
    PTVs = dict_images["Target"]

    # OARs
    list_OAR_names = [
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
    OAR_all = np.concatenate(
        [dict_images[OAR_name] for OAR_name in list_OAR_names], axis=0
    )

    # CT image
    CT = dict_images["CT"]
    CT = np.clip(CT, a_min=-1024, a_max=1500)
    CT = CT.astype(np.float32) / 1000.0

    # Possible mask
    possible_dose_mask = dict_images["Dose_Mask"]

    list_images = [
        np.concatenate((PTVs, OAR_all, CT), axis=0),  # Input
        possible_dose_mask,
    ]
    return list_images


def copy_sitk_imageinfo(image1, image2):
    image2.SetSpacing(image1.GetSpacing())
    image2.SetDirection(image1.GetDirection())
    image2.SetOrigin(image1.GetOrigin())

    return image2


# Input is C*Z*H*W
def flip_3d(input_, list_axes):
    if "Z" in list_axes:
        input_ = input_[:, ::-1, :, :]
    if "W" in list_axes:
        input_ = input_[:, :, :, ::-1]

    return input_


def test_time_augmentation(trainer, input_, TTA_mode):
    list_prediction_B = []

    for list_flip_axes in TTA_mode:
        # Do Augmentation before forward
        augmented_input = flip_3d(input_.copy(), list_flip_axes)
        augmented_input = torch.from_numpy(augmented_input.astype(np.float32))
        augmented_input = augmented_input.unsqueeze(0).to(trainer.setting.device)
        [_, prediction_B] = trainer.setting.network(augmented_input)

        # Aug back to original order
        prediction_B = flip_3d(
            np.array(prediction_B.cpu().data[0, :, :, :, :]), list_flip_axes
        )

        list_prediction_B.append(prediction_B[0, :, :, :])

    return np.mean(list_prediction_B, axis=0)
