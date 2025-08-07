# -*- encoding: utf-8 -*-
from deepdosesens.utils.data_utils import (
    read_data,
    pre_processing,
    test_time_augmentation,
    copy_sitk_imageinfo,
)
from deepdosesens.training.metrics import (
    get_Dose_score_and_DVH_score,
    get_Dose_score_and_DVH_score_per_ROI,
)
from deepdosesens.model.model import UNet
from deepdosesens.training.trainer import NetworkTrainer
import os
import sys
import json
import torch
import argparse
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

if os.path.abspath("..") not in sys.path:
    sys.path.insert(0, os.path.abspath(".."))


def inference(trainer, list_patient_dirs, save_path, do_TTA=True):
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    with torch.no_grad():
        trainer.setting.network.eval()
        for patient_dir in tqdm(list_patient_dirs):
            patient_id = patient_dir.split("/")[-1]

            dict_images = read_data(patient_dir)
            list_images = pre_processing(dict_images)

            input_ = list_images[0]
            possible_dose_mask = list_images[1]

            # Test-time augmentation
            if do_TTA:
                TTA_mode = [[], ["Z"], ["W"], ["Z", "W"]]
            else:
                TTA_mode = [[]]
            prediction = test_time_augmentation(trainer, input_, TTA_mode)

            # Pose-processing
            prediction[
                np.logical_or(possible_dose_mask[0, :, :, :] < 1, prediction < 0)
            ] = 0
            prediction = 70.0 * prediction

            # Save prediction to nii image
            templete_nii = sitk.ReadImage(patient_dir + "/Dose_Mask.nii.gz")
            prediction_nii = sitk.GetImageFromArray(prediction)
            prediction_nii = copy_sitk_imageinfo(templete_nii, prediction_nii)
            if not os.path.exists(save_path + "/" + patient_id):
                os.mkdir(save_path + "/" + patient_id)
            sitk.WriteImage(
                prediction_nii, save_path + "/" + patient_id + "/Dose.nii.gz"
            )


if __name__ == "__main__":

    root_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(root_dir, "results", "processed-dldp-UNet")
    data_dir = os.path.join(root_dir, "data", "processed-dldp")
    output_dir = model_dir
    os.makedirs(output_dir, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--GPU_id", type=int, default=-1, help="GPU id used for testing (default: 0)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.path.join(model_dir, "best_val_evaluation_index.pkl"),
    )
    parser.add_argument(
        "--TTA", type=bool, default=True, help="do test-time augmentation, default True"
    )
    args = parser.parse_args()

    trainer = NetworkTrainer()
    trainer.setting.project_name = "UNet"
    trainer.setting.output_dir = output_dir

    trainer.setting.network = UNet(
        in_ch=15,
        out_ch=1,
        list_ch=[-1, 16, 32, 64, 128, 256],
    )

    # Load model weights
    trainer.init_trainer(
        ckpt_file=args.model_path, list_GPU_ids=[args.GPU_id], only_network=True
    )

    dose_score = []
    dvh_score = []

    test_indices = [x for x in range(81, 101)]
    test_indices.append(x for x in range(109, 120))

    for subject_id in test_indices:
        # Start inference
        print("\n\n# Start inference !")
        list_patient_dirs = [os.path.join(data_dir, "DLDP_" + str(subject_id).zfill(3))]
        inference(
            trainer,
            list_patient_dirs,
            save_path=os.path.join(trainer.setting.output_dir, "Prediction"),
            do_TTA=args.TTA,
        )

        # Evaluation
        print("\n\n# Start evaluation !")
        Dose_score, DVH_score = get_Dose_score_and_DVH_score_per_ROI(
            prediction_dir=os.path.join(trainer.setting.output_dir, "Prediction"),
            patient_id=subject_id,
            gt_dir=data_dir,
        )

        with open(
            trainer.setting.output_dir
            + "/Prediction/"
            + "DLDP_"
            + str(subject_id).zfill(3)
            + "/dose_score.json",
            "w",
        ) as fp:
            json.dump(Dose_score, fp)

        dose_df = pd.DataFrame.from_dict(
            Dose_score["DLDP_" + str(subject_id).zfill(3)], orient="index"
        )
        dose_df.to_csv(
            trainer.setting.output_dir
            + "/Prediction/"
            + "DLDP_"
            + str(subject_id).zfill(3)
            + "/dose_score.csv"
        )

        with open(
            trainer.setting.output_dir
            + "/Prediction/"
            + "DLDP_"
            + str(subject_id).zfill(3)
            + "/dvh_score.json",
            "w",
        ) as fp:
            json.dump(DVH_score, fp)

        dvh_df = pd.DataFrame.from_dict(
            DVH_score["DLDP_" + str(subject_id).zfill(3)], orient="index"
        )
        dvh_df.to_csv(
            trainer.setting.output_dir
            + "/Prediction/"
            + "DLDP_"
            + str(subject_id).zfill(3)
            + "/dvh_score.csv"
        )

        Dose_score, DVH_score = get_Dose_score_and_DVH_score(
            prediction_dir=os.path.join(trainer.setting.output_dir, "Prediction"),
            patient_id=subject_id,
            gt_dir=data_dir,
        )

        dose_score.append(Dose_score)
        dvh_score.append(DVH_score)
        print("\n\nDose score is: " + str(Dose_score))
        print("DVH score is: " + str(DVH_score))

    print(
        "Mean dose score: "
        + str(np.mean(dose_score))
        + " ("
        + str(np.std(dose_score))
        + ")"
    )
    print(
        "Mean dvh score: "
        + str(np.mean(dvh_score))
        + " ("
        + str(np.std(dvh_score))
        + ")"
    )
