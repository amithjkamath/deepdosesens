# -*- encoding: utf-8 -*-
import os
import json
import sys
import argparse
import pandas as pd

if os.path.abspath("..") not in sys.path:
    sys.path.insert(0, os.path.abspath(".."))

from utils.data_utils import read_data, pre_processing, test_time_augmentation, copy_sitk_imageinfo
from validation.evaluate_DLDP import *
from model.C3D.model import Model
from training.network_trainer import *


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
            templete_nii = sitk.ReadImage(patient_dir + "/Dose_Mask_resized.nii.gz")
            prediction_nii = sitk.GetImageFromArray(prediction)
            prediction_nii = copy_sitk_imageinfo(templete_nii, prediction_nii)
            if not os.path.exists(save_path + "/" + patient_id):
                os.mkdir(save_path + "/" + patient_id)
            sitk.WriteImage(
                prediction_nii, save_path + "/" + patient_id + "/Dose_resized.nii.gz"
            )


if __name__ == "__main__":

    root_dir = "/Users/amithkamath/repo"
    model_dir = os.path.join(root_dir, "deep-planner/models/")
    output_dir = os.path.join(root_dir, "deep-planner/output_perROI")
    os.makedirs(output_dir, exist_ok=True)

    gt_dir = os.path.join(root_dir, "deep-planner/data/resized")
    test_dir = gt_dir # change this if somewhere else.

    if not os.path.exists(model_dir):
        raise Exception(
            "OpenKBP_C3D should be prepared before testing, please run prepare_OpenKBP_C3D.py"
        )

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
    trainer.setting.project_name = "C3D"
    trainer.setting.output_dir = output_dir

    trainer.setting.network = Model(
        in_ch=15,
        out_ch=1,
        list_ch_A=[-1, 16, 32, 64, 128, 256],
        list_ch_B=[-1, 32, 64, 128, 256, 512],
    )

    # Load model weights
    trainer.init_trainer(
        ckpt_file=args.model_path, list_GPU_ids=[args.GPU_id], only_network=True
    )

    for subject_id in [81, 82]:

        # Start inference
        print("\n\n# Start inference !")
        list_patient_dirs = [
            os.path.join(test_dir, "DLDP_" + str(subject_id).zfill(3))
        ]
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
            gt_dir=gt_dir,
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
