# DeepDoseSens: Evaluating Sensitivity of Deep Learning-Based Radiotherapy Dose Prediction to Organs-at-Risk Segmentation Variability

![ISBI 2023](https://img.shields.io/badge/Conference-ISBI%202023-blue)

This repository accompanies our paper:

**"How Sensitive Are Deep Learning Based Radiotherapy Dose Prediction Models To Variability In Organs At Risk Segmentation?"**  
Accepted at the 20th International Symposium on Biomedical Imaging (ISBI), 2023.  

**Authors:** Amith Kamath, Robert Poel, Jonas Willmann, Nicolaus Andratschke, Mauricio Reyes

See a short video description of this work here:

[<img src="https://i.ytimg.com/vi/Lz5-n4lA3QM/maxresdefault.jpg" width="50%">](https://youtu.be/Lz5-n4lA3QM "Sensitivity of Deep Learning dose Prediction models")

🔗 [Project Website](https://amithjkamath.github.io/projects/2023-isbi-deepdosesens/)

---

## Overview

This project investigates the robustness of deep learning models for radiotherapy dose prediction in glioblastoma patients, focusing on how variability in organs-at-risk (OAR) segmentation affects model performance. We introduce a controlled perturbation framework to simulate realistic segmentation variations and assess their impact on dose prediction accuracy.

---

## Key Contributions

- **Controlled Perturbation Framework:** Developed a method to simulate realistic variations in OAR segmentations.
- **Robustness Assessment:** Analyzed how segmentation variability influences dose prediction.
- **Model Comparison:** Evaluated multiple deep learning models for robustness against OAR perturbations.

---

## Methodology

- **Data:** Glioblastoma patient CT scans, OAR segmentations, and corresponding dose distributions.
- **Perturbation Techniques:** Applied geometric deformations and noise to mimic segmentation variability.
- **Model Training:** Used CNN-based architectures trained on original segmentations and tested on perturbed data.
- **Metrics:** Evaluated using Mean Absolute Error (MAE), Dose-Volume Histogram (DVH) differences, and other relevant metrics.

---

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- MONAI
- NumPy
- SciPy
- Matplotlib

### Installation

```bash
git clone https://github.com/amithjkamath/deepdosesens.git
cd deepdosesens
pip install -r requirements.txt
```

If this is useful in your research, please consider citing:

    @inproceedings{kamath2023doseprediction,
    title={How sensitive are deep learning based radiotherapy dose prediction models to variability in Organs At Risk segmentation?},
    author={Kamath, Amith and Poel, Robert and Willmann, Jonas and Andratschke, Nicolaus and Reyes, Mauricio},
    booktitle={2023 IEEE 20th International Symposium on Biomedical Imaging (ISBI)},
    pages={1--4},
    year={2023},
    organization={IEEE}
    }

## Credits
Major props to the code and organization in https://github.com/LSL000UD/RTDosePrediction, which is what this model is based on (looks like this repo is not maintained/available anymore!)
