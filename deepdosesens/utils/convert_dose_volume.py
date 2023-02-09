"""
Reading CT and RTSS data from .dcm files
"""

import glob
import os
from tqdm import tqdm

import pydicom
import pymedphys
import SimpleITK as sitk


def rtdose_to_nifti(base_input_path, base_output_path, n_subjects):
    """
    RTDOSE_TO_NIFTI converts RD*.dcm RT Dose files to NIfTI volumes.
    """
    for subject_id in tqdm(range(1, n_subjects + 1)):
        try:
            str_id = str(subject_id).zfill(3)
            subject_name = "DLDP_" + str_id
            fpath = os.path.join(base_input_path, subject_name)

            rtdose_file = glob.glob(fpath + "//RD*.dcm")
            # print("Analyzing subject: " + subject_name)
            ds = pydicom.dcmread(rtdose_file[0])
            dose_image_sitk = sitk.ReadImage(rtdose_file[0])
            (dose_axes, dose_array) = pymedphys.dicom.zyx_and_dose_from_dataset(ds)
            dose_image = sitk.GetImageFromArray(dose_array)
            dose_image.CopyInformation(dose_image_sitk)

            ct_file = os.path.join(base_output_path, subject_name, "CT.nii.gz")
            ct_image = sitk.ReadImage(ct_file)

            resample = sitk.ResampleImageFilter()
            resample.SetInterpolator = sitk.sitkLinear
            resample.SetOutputDirection = ct_image.GetDirection()
            resample.SetOutputOrigin(ct_image.GetOrigin())
            resample.SetOutputSpacing(ct_image.GetSpacing())
            resample.SetSize(ct_image.GetSize())

            new_dose_image = resample.Execute(dose_image)
            subject_output_path = os.path.join(base_output_path, subject_name)
            sitk.WriteImage(
                new_dose_image, os.path.join(subject_output_path, "Dose.nii.gz")
            )
            # print("Completed subject: " + subject_name)
        except Exception as ex:
            print(ex)
            print("Errored on subject: ", subject_name, "; skipping it.")


if __name__ == "__main__":
    input_path = "/home/akamath/Documents/deep-planner/data/raw-dldp-old/"
    output_path = "/home/akamath/Documents/deep-planner/data/interim-dldp-dt/"
    num_subjects = 100
    rtdose_to_nifti(input_path, output_path, num_subjects)
