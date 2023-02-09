"""
Reading CT and RTSS data from .dcm files.
See https://github.com/brianmanderson/Dicom_RT_and_Images_to_Mask/blob/main/Examples/DICOMRTTool_Tutorial.ipynb for more.
"""

import os
from tqdm import tqdm

from DicomRTTool.ReaderWriter import DicomReaderWriter
import SimpleITK as sitk


def ct_to_nifti(base_input_path, base_output_path, subject_name, out_fname):
    """
    CT_TO_NIFTI converts CT DICOM volumes to NIfTI.
    """

    fpath = os.path.join(base_input_path, subject_name)
    dicom_reader = DicomReaderWriter(description="Examples", arg_max=True)
    dicom_reader.walk_through_folders(fpath)
    # all_rois = dicom_reader.return_rois(print_rois=True)

    dicom_reader.set_contour_names_and_associations(
        Contour_Names=["Brain"], associations={"brain": "Brain"}
    )
    indexes = (
        dicom_reader.which_indexes_have_all_rois()
    )  # Check to see which indexes have all of the rois we want, now we can see indexes
    pt_indx = indexes[-1]
    dicom_reader.set_index(
        pt_indx
    )  # This index has all the structures, corresponds to pre-RT T1-w image for patient 011
    dicom_reader.get_images_and_mask()  # Load up the images and mask for the requested index

    image = dicom_reader.ArrayDicom  # image array
    mask = dicom_reader.mask  # mask array

    out_path = os.path.join(base_output_path, subject_name)
    os.makedirs(out_path, exist_ok=True)
    image_sitk_handle = dicom_reader.dicom_handle
    sitk.WriteImage(image_sitk_handle, os.path.join(out_path, out_fname))
    return


def rtss_to_nifti(
    base_input_path,
    base_output_path,
    subject_name,
    out_fname,
    contour_names,
    associations,
):
    """
    RTSS_TO_NIFTI converts RTSS DICOM volumes to NIfTI.
    """
    fpath = os.path.join(base_input_path, subject_name)
    # rtstruct_file = glob.glob(fpath + "//RS*.dcm")[0]
    dicom_reader = DicomReaderWriter(description="Examples", arg_max=True)
    dicom_reader.walk_through_folders(fpath)
    # all_rois = dicom_reader.return_rois(print_rois=True)

    dicom_reader.set_contour_names_and_associations(
        Contour_Names=contour_names, associations=associations
    )
    indexes = (
        dicom_reader.which_indexes_have_all_rois()
    )  # Check to see which indexes have all of the rois we want, now we can see indexes
    pt_indx = indexes[-1]
    dicom_reader.set_index(
        pt_indx
    )  # This index has all the structures, corresponds to pre-RT T1-w image for patient 011
    dicom_reader.get_images_and_mask()  # Load up the images and mask for the requested index

    image = dicom_reader.ArrayDicom  # image array
    mask = dicom_reader.mask  # mask array

    out_path = os.path.join(base_output_path, subject_name)
    os.makedirs(out_path, exist_ok=True)
    mask_sitk_handle = dicom_reader.annotation_handle  # SimpleITK mask handle
    sitk.WriteImage(mask_sitk_handle, os.path.join(out_path, out_fname))
    return


def rt_to_nifti(input_folder, output_folder, n_subjects):
    """
    RT_TO_NIFTI converts RT data in base_input_folder to NIfTI in base_output_folder.
    """

    for subject_id in tqdm(range(1, n_subjects + 1)):
        str_id = str(subject_id).zfill(3)
        subject_name = "DLDP_" + str_id

        try:
            ct_to_nifti(input_folder, output_folder, subject_name, "CT.nii.gz")

            contour_names = ["PTV"]
            associations = {"ptv_high": "PTV", "ptv_test": "PTV"}
            rtss_to_nifti(
                input_folder,
                output_folder,
                subject_name,
                "Target.nii.gz",
                contour_names,
                associations,
            )

            contour_names = ["Brain"]
            associations = {"brain": "Brain"}
            rtss_to_nifti(
                input_folder,
                output_folder,
                subject_name,
                "Brain.nii.gz",
                contour_names,
                associations,
            )

            contour_names = ["Body"]
            associations = {"body": "Body"}
            rtss_to_nifti(
                input_folder,
                output_folder,
                subject_name,
                "Dose_Mask.nii.gz",
                contour_names,
                associations,
            )

            contour_names = ["Brainstem"]
            associations = {"r_brainstem": "BrainStem", "brainstem": "BrainStem"}
            rtss_to_nifti(
                input_folder,
                output_folder,
                subject_name,
                "BrainStem.nii.gz",
                contour_names,
                associations,
            )

            contour_names = ["Cochlea_L"]
            associations = {"r_cochlea_l": "Cochlea_L", "cochlea_l": "Cochlea_L"}
            rtss_to_nifti(
                input_folder,
                output_folder,
                subject_name,
                "Cochlea_L.nii.gz",
                contour_names,
                associations,
            )

            contour_names = ["Cochlea_R"]
            associations = {"r_cochlea_r": "Cochlea_R", "cochlea_r": "Cochlea_R"}
            rtss_to_nifti(
                input_folder,
                output_folder,
                subject_name,
                "Cochlea_R.nii.gz",
                contour_names,
                associations,
            )

            contour_names = ["Eye_L"]
            associations = {"r_eye_l": "Eye_L", "eye_l": "Eye_L"}
            rtss_to_nifti(
                input_folder,
                output_folder,
                subject_name,
                "Eye_L.nii.gz",
                contour_names,
                associations,
            )

            contour_names = ["Eye_R"]
            associations = {"r_eye_r": "Eye_R", "eye_r": "Eye_R"}
            rtss_to_nifti(
                input_folder,
                output_folder,
                subject_name,
                "Eye_R.nii.gz",
                contour_names,
                associations,
            )

            contour_names = ["Hippocampus_L"]
            associations = {
                "r_hippocampus_l": "Hippocampus_L",
                "hippocampus_l": "Hippocampus_L",
            }
            rtss_to_nifti(
                input_folder,
                output_folder,
                subject_name,
                "Hippocampus_L.nii.gz",
                contour_names,
                associations,
            )

            contour_names = ["Hippocampus_R"]
            associations = {
                "r_hippocampus_r": "Hippocampus_R",
                "hippocampus_r": "Hippocampus_R",
            }
            rtss_to_nifti(
                input_folder,
                output_folder,
                subject_name,
                "Hippocampus_R.nii.gz",
                contour_names,
                associations,
            )

            contour_names = ["LacrimalGland_L"]
            associations = {
                "r_lacrimal_l": "LacrimalGland_L",
                "lacrimal_l": "LacrimalGland_L",
            }
            rtss_to_nifti(
                input_folder,
                output_folder,
                subject_name,
                "LacrimalGland_L.nii.gz",
                contour_names,
                associations,
            )

            contour_names = ["LacrimalGland_R"]
            associations = {
                "r_lacrimal_r": "LacrimalGland_R",
                "lacrimal_r": "LacrimalGland_R",
            }
            rtss_to_nifti(
                input_folder,
                output_folder,
                subject_name,
                "LacrimalGland_R.nii.gz",
                contour_names,
                associations,
            )

            contour_names = ["Chiasm"]
            associations = {"r_opticchiasm": "Chiasm", "opticchiasm": "Chiasm"}
            rtss_to_nifti(
                input_folder,
                output_folder,
                subject_name,
                "Chiasm.nii.gz",
                contour_names,
                associations,
            )

            contour_names = ["OpticNerve_L"]
            associations = {
                "r_opticenerve_l": "OpticNerve_L",
                "r_opticnerve_l": "OpticNerve_L",
                "opticenerve_l": "OpticNerve_L",
                "opticnerve_l": "OpticNerve_L",
            }
            rtss_to_nifti(
                input_folder,
                output_folder,
                subject_name,
                "OpticNerve_L.nii.gz",
                contour_names,
                associations,
            )

            contour_names = ["OpticNerve_R"]
            associations = {
                "r_opticnerve_r": "OpticNerve_R",
                "opticnerve_r": "OpticNerve_R",
            }
            rtss_to_nifti(
                input_folder,
                output_folder,
                subject_name,
                "OpticNerve_R.nii.gz",
                contour_names,
                associations,
            )

            contour_names = ["Pituitary"]
            associations = {"r_pituitary": "Pituitary", "pituitary": "Pituitary"}
            rtss_to_nifti(
                input_folder,
                output_folder,
                subject_name,
                "Pituitary.nii.gz",
                contour_names,
                associations,
            )

        except Exception as ex:
            print(ex)
            print("Errored on subject: ", subject_name, "; skipping it.")


if __name__ == "__main__":
    input_path = "/home/akamath/Documents/deep-planner/data/raw-dldp-old/"
    output_path = "/home/akamath/Documents/deep-planner/data/interim-dldp-dt/"
    num_subjects = 100
    rt_to_nifti(input_path, output_path, num_subjects)
