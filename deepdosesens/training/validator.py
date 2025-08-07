# -*- encoding: utf-8 -*-
import torch
import numpy as np
from deepdosesens.data.dataloader import val_transform, read_data, pre_processing
from deepdosesens.training.metrics import compute_abs_dose_difference


def validate(trainer, list_patient_dirs):
    """Validation of the model on the given patient directories.
    Computes the absolute dose difference for each patient and logs the results.
    Returns the mean dose score across all patients, to be used for model evaluation.
    """
    dose_scores = []

    with torch.no_grad():
        trainer.setting.network.eval()
        for patient_dir in list_patient_dirs:
            patient_name = patient_dir.split("/")[-1]

            dict_images = read_data(patient_dir)
            list_images = pre_processing(dict_images)

            input_ = list_images[0]
            reference_dose_ = list_images[1]
            dose_mask_ = list_images[2]

            # Forward
            [input_] = val_transform([input_])
            input_ = input_.unsqueeze(0).to(trainer.setting.device)
            if trainer.setting.project_name == "C3D":
                [_, prediction] = trainer.setting.network(input_)
            elif trainer.setting.project_name == "UNet":
                prediction = trainer.setting.network(input_)
            else:
                raise ValueError(
                    f"The trainer project name {trainer.setting.project_name} is unrecognized."
                )

            prediction = np.array(prediction.cpu().data[0, :, :, :, :])
            # Post processing and evaluation
            prediction[np.logical_or(dose_mask_ < 1, prediction < 0)] = 0
            dose_score = 70.0 * compute_abs_dose_difference(
                prediction.squeeze(0),
                reference_dose_.squeeze(0),
                dose_mask_.squeeze(0),
            )
            dose_scores.append(dose_score)

            trainer.print_log_to_file(
                "=> " + patient_name + ":  " + str(dose_score), "a"
            )

    trainer.print_log_to_file(
        "=> mean Dose score: " + str(np.mean(dose_scores)),
        "a",
    )
    # Evaluation score is the higher the better
    return -np.mean(dose_scores)
