# -*- encoding: utf-8 -*-
import os
import sys
import logging
import numpy as np

from deepdosesens.training.metrics import compute_scores

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if os.path.abspath("..") not in sys.path:
    sys.path.insert(0, os.path.abspath(".."))


if __name__ == "__main__":

    root_dir = os.path.dirname(os.path.abspath(__file__))
    reference_path = os.path.join(root_dir, "data", "processed-dldp")
    prediction_path = os.path.join(
        root_dir, "results", "processed-dldp-UNet", "Prediction"
    )

    dose_scores = {}
    dvh_scores = {}

    structure_list = [
        "Brain",
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
    ]

    test_subjects = os.listdir(prediction_path)

    for subject_name in test_subjects:
        logger.info("Start evaluation for subject %s ...", subject_name)

        prediction_dir = os.path.join(prediction_path, subject_name)
        reference_dir = os.path.join(reference_path, subject_name)
        dose_score, dvh_score = compute_scores(
            prediction_path=prediction_dir,
            reference_path=reference_dir,
            structure_list=structure_list,
        )

        dose_scores[subject_name] = dose_score
        dvh_scores[subject_name] = dvh_score
        structure_list = list(dose_score.keys())
        for structure_name in structure_list:
            logger.info(
                "Dose score for %s, %s: %.4f",
                subject_name,
                structure_name,
                dose_score[structure_name],
            )
        structure_list = list(dvh_score.keys())
        for structure_name in structure_list:
            logger.info(
                "DVH score for %s, %s: %.4f",
                subject_name,
                structure_name,
                dvh_score[structure_name],
            )

    logger.info("Evaluation completed for all subjects.")

    for structure in structure_list:
        aggregate_dose_scores = list(
            dose_scores[item][structure] for item in dose_scores
        )
        logger.info(
            "Mean(sd) %s dose score over all subjects: %.4f (%.4f)",
            structure,
            np.mean(aggregate_dose_scores),
            np.std(aggregate_dose_scores),
        )
