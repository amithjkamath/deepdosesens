"""
Resample volumes to a consistent space for training the C3D network
"""

import os
from tqdm import tqdm
import numpy as np
import SimpleITK as sitk


def resize_volume(ref_img, output_size, is_label=False):
    """
    RESIZE_VOLUME resizes volumes using sitk.
    See https://gist.github.com/zivy/79d7ee0490faee1156c1277a78e4a4c4 for more.
    Physical image size corresponds to the largest physical size in the
    training set, or any other arbitrary size.
    """
    reference_physical_size = np.zeros(3)
    reference_physical_size[:] = [
        (sz - 1) * spc if sz * spc > mx else mx
        for sz, spc, mx in zip(
            ref_img.GetSize(), ref_img.GetSpacing(), reference_physical_size
        )
    ]

    # Create the reference image with a zero origin, identity direction cosine matrix and dimension
    reference_origin = np.zeros(3)
    reference_direction = np.identity(3).flatten()
    reference_size = output_size
    reference_spacing = [
        phys_sz / (sz - 1)
        for sz, phys_sz in zip(reference_size, reference_physical_size)
    ]

    reference_image = sitk.Image(reference_size, ref_img.GetPixelIDValue())
    reference_image.SetOrigin(reference_origin)
    reference_image.SetSpacing(reference_spacing)
    reference_image.SetDirection(reference_direction)

    # Always use the TransformContinuousIndexToPhysicalPoint to compute an indexed point's physical coordinates as
    # this takes into account size, spacing and direction cosines. For the vast majority of images the direction
    # cosines are the identity matrix, but when this isn't the case simply multiplying the central index by the
    # spacing will not yield the correct coordinates resulting in a long debugging session.
    reference_center = np.array(
        reference_image.TransformContinuousIndexToPhysicalPoint(
            np.array(reference_image.GetSize()) / 2.0
        )
    )

    # origins to each other.
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(ref_img.GetDirection())
    transform.SetTranslation(np.array(ref_img.GetOrigin()) - reference_origin)
    # Modify the transformation to align the centers of the original and reference image instead of their origins.
    centering_transform = sitk.TranslationTransform(3)
    img_center = np.array(
        ref_img.TransformContinuousIndexToPhysicalPoint(
            np.array(ref_img.GetSize()) / 2.0
        )
    )
    centering_transform.SetOffset(
        np.array(transform.GetInverse().TransformPoint(img_center) - reference_center)
    )
    centered_transform = sitk.CompositeTransform(transform)
    centered_transform.AddTransform(centering_transform)
    # Using the linear interpolator as these are intensity images, if there is a need to resample a ground truth
    # segmentation then the segmentation image should be resampled using the NearestNeighbor interpolator so that
    # no new labels are introduced.
    if is_label:
        out_img = sitk.Resample(
            ref_img, reference_image, centered_transform, sitk.sitkNearestNeighbor, 0.0
        )
    else:
        out_img = sitk.Resample(
            ref_img, reference_image, centered_transform, sitk.sitkLinear, 0.0
        )
    return out_img


def resize_nifti_volume(input_path, output_fname, output_size, is_label):
    """
    RESIZE_NIFTI_VOLUME resizes NIfTI volumes using the resize_volume method.
    """

    # Import the reference image
    ref_img_reader = sitk.ImageFileReader()
    ref_img_reader.SetFileName(input_path)
    ref_img = ref_img_reader.Execute()

    out_img = resize_volume(ref_img, output_size, is_label)
    sitk.WriteImage(out_img, output_fname, True)
    return


def resize_to(base_input_path, base_output_path, n_subjects, output_size):
    """
    RESIZE_TO resizes OAR volumes and CT volumes to output_size.
    """

    for subject_id in tqdm(range(1, n_subjects + 1)):
        try:
            str_id = str(subject_id).zfill(3)
            subject_name = "DLDP_" + str_id

            # print("Analyzing subject: ", subject_name)
            os.makedirs(os.path.join(base_output_path, subject_name), exist_ok=True)

            ct_file = os.path.join(base_input_path, subject_name, "CT.nii.gz")
            output_fname = os.path.join(base_output_path, subject_name, "CT.nii.gz")
            resize_nifti_volume(ct_file, output_fname, output_size, is_label=False)
            # os.remove(ct_file)

            dose_file = os.path.join(base_input_path, subject_name, "Dose.nii.gz")
            output_fname = os.path.join(base_output_path, subject_name, "Dose.nii.gz")
            resize_nifti_volume(dose_file, output_fname, output_size, is_label=False)
            # os.remove(dose_file)

            labels = [
                "Brain",
                "BrainStem",
                "Chiasm",
                "Cochlea_L",
                "Cochlea_R",
                "Dose_Mask",
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

            for label in labels:
                label_file = os.path.join(
                    base_input_path, subject_name, label + ".nii.gz"
                )
                output_fname = os.path.join(
                    base_output_path, subject_name, label + ".nii.gz"
                )
                resize_nifti_volume(
                    label_file, output_fname, output_size, is_label=True
                )
                # os.remove(label_file)

            # print("Completed subject: ", subject_name)

        except Exception as ex:
            print(ex)
            print("Errored on subject: ", subject_name, "; skipping it.")


if __name__ == "__main__":

    input_path = "/home/akamath/Documents/deep-planner/data/interim-dldp/"
    output_path = "/home/akamath/Documents/deep-planner/data/processed-dldp/"
    num_subjects = 100
    output_dim = [128, 128, 128]
    resize_to(input_path, output_path, num_subjects, output_dim)
