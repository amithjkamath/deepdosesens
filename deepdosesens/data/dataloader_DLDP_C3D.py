# -*- encoding: utf-8 -*-
import torch.utils.data as data
import os
import SimpleITK as sitk
import numpy as np
import random
import cv2

from data.augmentation_OpenKBP_C3D import (
    random_flip_3d,
    random_rotate_around_z_axis,
    random_translate,
    to_tensor,
)

"""
images are always C*Z*H*W
"""


def read_data(patient_dir):
    dict_images = {}
    list_structures = [
        "CT",
        "Dose_Mask",
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
        "Dose",
        "Target",
    ]

    for structure_name in list_structures:
        structure_file = patient_dir + "/" + structure_name + ".nii.gz"

        if structure_name == "CT":
            dtype = sitk.sitkInt16
        elif structure_name == "Dose":
            dtype = sitk.sitkFloat32
        else:
            dtype = sitk.sitkUInt8

        if os.path.exists(structure_file):
            dict_images[structure_name] = sitk.ReadImage(structure_file, dtype)
            # To numpy array (C * Z * H * W)
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

    # Dose image
    dose = dict_images["Dose"] / 70.0

    # Possible_dose_mask, the region that can receive dose
    possible_dose_mask = dict_images["Dose_Mask"]

    list_images = [
        np.concatenate((PTVs, OAR_all, CT), axis=0),  # Input
        dose,  # Label
        possible_dose_mask,
    ]
    return list_images


def train_transform(list_images):
    # list_images = [Input, Label(gt_dose), possible_dose_mask]
    # Random flip
    list_images = random_flip_3d(list_images, list_axis=(0, 2), p=0.8)

    # Random rotation
    list_images = random_rotate_around_z_axis(
        list_images,
        list_angles=(0, 40, 80, 120, 160, 200, 240, 280, 320),
        list_boder_value=(0, 0, 0),
        list_interp=(cv2.INTER_NEAREST, cv2.INTER_NEAREST, cv2.INTER_NEAREST),
        p=0.3,
    )

    """
    # Random translation, but make use the region can receive dose is remained
    list_images = random_translate(
        list_images,
        roi_mask=list_images[2][0, :, :, :],  # the possible dose mask
        p=0.8,
        max_shift=20,
        list_pad_value=[0, 0, 0],
    )
    """
    # To torch tensor
    list_images = to_tensor(list_images)
    return list_images


def val_transform(list_images):
    list_images = to_tensor(list_images)
    return list_images


class MyDataset(data.Dataset):
    def __init__(self, data_paths, num_samples_per_epoch, phase):
        # 'train' or 'val'
        self.data_paths = data_paths
        self.phase = phase
        self.num_samples_per_epoch = num_samples_per_epoch
        self.transform = {"train": train_transform, "val": val_transform}

        self.list_case_id = self.data_paths[phase]

        random.shuffle(self.list_case_id)
        self.sum_case = len(self.list_case_id)

    def __getitem__(self, index_):
        if index_ <= self.sum_case - 1:
            case_id = self.list_case_id[index_]
        else:
            new_index_ = index_ - (index_ // self.sum_case) * self.sum_case
            case_id = self.list_case_id[new_index_]

        dict_images = read_data(case_id)
        list_images = pre_processing(dict_images)

        list_images = self.transform[self.phase](list_images)
        return list_images

    def __len__(self):
        return self.num_samples_per_epoch


def get_loader(
    data_paths,
    train_bs=1,
    val_bs=1,
    train_num_samples_per_epoch=1,
    val_num_samples_per_epoch=1,
    num_works=0,
):
    train_dataset = MyDataset(
        data_paths, num_samples_per_epoch=train_num_samples_per_epoch, phase="train"
    )
    val_dataset = MyDataset(
        data_paths, num_samples_per_epoch=val_num_samples_per_epoch, phase="val"
    )

    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=train_bs,
        shuffle=True,
        num_workers=num_works,
        pin_memory=False,
    )
    val_loader = data.DataLoader(
        dataset=val_dataset,
        batch_size=val_bs,
        shuffle=False,
        num_workers=num_works,
        pin_memory=False,
    )

    return train_loader, val_loader
