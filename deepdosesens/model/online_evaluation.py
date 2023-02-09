# -*- encoding: utf-8 -*-
from data.dataloader_DLDP_C3D import val_transform, read_data, pre_processing
from validation.evaluate_DLDP import *
from model import *
import torch


def online_evaluation(trainer, list_patient_dirs):

    list_Dose_score = []

    with torch.no_grad():
        trainer.setting.network.eval()
        for patient_dir in list_patient_dirs:
            patient_name = patient_dir.split("/")[-1]

            dict_images = read_data(patient_dir)
            list_images = pre_processing(dict_images)

            input_ = list_images[0]
            gt_dose = list_images[1]
            possible_dose_mask = list_images[2]

            # Forward
            [input_] = val_transform([input_])
            input_ = input_.unsqueeze(0).to(trainer.setting.device)
            [_, prediction_B] = trainer.setting.network(input_)
            prediction_B = np.array(prediction_B.cpu().data[0, :, :, :, :])

            # Post processing and evaluation
            prediction_B[np.logical_or(possible_dose_mask < 1, prediction_B < 0)] = 0
            Dose_score = 70.0 * get_3D_Dose_dif(
                prediction_B.squeeze(0),
                gt_dose.squeeze(0),
                possible_dose_mask.squeeze(0),
            )
            list_Dose_score.append(Dose_score)

            try:
                trainer.print_log_to_file(
                    "========> " + patient_name + ":  " + str(Dose_score), "a"
                )
            except:
                pass

    try:
        trainer.print_log_to_file(
            "===============================================> mean Dose score: "
            + str(np.mean(list_Dose_score)),
            "a",
        )
    except:
        pass
    # Evaluation score is the higher the better
    return -np.mean(list_Dose_score)
