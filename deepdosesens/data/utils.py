import os
import numpy as np
import SimpleITK as sitk
import torch


def read_nifti_image(file_path: str, type=sitk.sitkFloat32) -> np.ndarray:
    """Read a NIfTI image and return it as a numpy array."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    image = sitk.ReadImage(file_path, type)
    image_array = sitk.GetArrayFromImage(image)
    return image_array


def get_spacing(file_path: str) -> tuple:
    """Get the spacing of a NIfTI image."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    image = sitk.ReadImage(file_path)
    return image.GetSpacing()


def duplicate_image_metadata(
    source_image: sitk.Image, target_image: sitk.Image
) -> sitk.Image:
    """Duplicate metadata from source_image to target_image.
    This includes spacing, direction, and origin.
    """
    target_image.SetSpacing(source_image.GetSpacing())
    target_image.SetDirection(source_image.GetDirection())
    target_image.SetOrigin(source_image.GetOrigin())

    return target_image


def read_data(patient_dir):
    """Read data from the patient directory.
    Returns a dictionary with images loaded from the specified files.
    Each key corresponds to a structure name, and the value is a numpy array.
    If a file does not exist, it returns a zero-filled array of shape (1, 128, 128, 128).
    """
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
    """Pre-process the images for inference.
    Combines PTVs, OARs, and CT images into a single input array.
    Returns a list containing the input array and a possible dose mask.
    """
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


# Input is C*Z*H*W
def flip_3d(input_, list_axes):
    if "Z" in list_axes:
        input_ = input_[:, ::-1, :, :]
    if "W" in list_axes:
        input_ = input_[:, :, :, ::-1]

    return input_


def test_time_augmentation(trainer, input_, TTA_mode):
    list_prediction = []

    for list_flip_axes in TTA_mode:
        # Do Augmentation before forward
        augmented_input = flip_3d(input_.copy(), list_flip_axes)
        augmented_input = torch.from_numpy(augmented_input.astype(np.float32))
        augmented_input = augmented_input.unsqueeze(0).to(trainer.setting.device)
        if trainer.setting.project_name == "C3D":
            [_, prediction] = trainer.setting.network(augmented_input)
        elif trainer.setting.project_name == "UNet":
            prediction = trainer.setting.network(augmented_input)

        # Aug back to original order
        prediction = flip_3d(
            np.array(prediction.cpu().data[0, :, :, :, :]), list_flip_axes
        )

        list_prediction.append(prediction[0, :, :, :])

    return np.mean(list_prediction, axis=0)
