# DeepDoseSens: dose prediction for glioblastoma, and how sensitive it is to contour changes

![ISBI 2023](https://img.shields.io/badge/Conference-ISBI%202023-blue) ![Cancers 2023](https://img.shields.io/badge/Journal-Cancers%202023-green) ![Python](https://img.shields.io/badge/python-3.9%2B-blue) ![License](https://img.shields.io/github/license/amithjkamath/deepdosesens)

This repository accompanies **two papers**:

- **ISBI 2023** — *How sensitive are deep learning based radiotherapy dose
  prediction models to variability in Organs At Risk segmentation?*
  Amith Kamath, Robert Poel, Jonas Willmann, Nicolaus Andratschke, Mauricio Reyes
- **Cancers 2023** — *Deep-Learning-Based Dose Predictor for Glioblastoma —
  Assessing the Sensitivity and Robustness for Dose Awareness in Contouring*
  Robert Poel, Amith Kamath, Jonas Willmann, Nicolaus Andratschke, Ekin Ermiş,
  Daniel M. Aebersold, Peter Manser, Mauricio Reyes

Data, model weights and predicted dose volumes are archived outside this repository
and shared on request — see [Configuring where the data lives](#configuring-where-the-data-lives).

See a short video description of this work here:

[<img src="https://i.ytimg.com/vi/Lz5-n4lA3QM/maxresdefault.jpg" width="50%">](https://youtu.be/Lz5-n4lA3QM "Sensitivity of Deep Learning dose Prediction models")

🔗 [Project Website](https://amithjkamath.github.io/projects/2023-isbi-deepdosesens/)

---

## Overview

A cascaded 3D U-Net predicts the full 3D dose distribution for glioblastoma VMAT
treatment from a planning CT and the target and organ-at-risk contours, in seconds
instead of the hours a planning system needs. The question both papers ask is
whether such a model is **sensitive enough to be useful for contour quality
assurance**: if an organ contour is edited, does the predicted dose move the way the
re-optimised clinical plan does? If it does, a contour can be judged by its
dosimetric consequence rather than by geometry alone.

- **ISBI 2023** establishes the model's accuracy and its sensitivity to ten
  plausible left optic nerve contours: predicted and planned dose differences
  correlate at 0.926, while the Dice coefficient correlates at only −0.471. Dice
  says a contour is different; only the dose says whether that matters.
- **Cancers 2023** stress-tests the same model on out-of-distribution target shapes
  — concave targets and targets split into several lesions — and shows that adding
  six such cases to the training set recovers the conformity it loses.

---

## Demonstration videos

`docs/videos/` holds one clip per experiment. See
[README-videos.md](deepdosesens/visualization/README-videos.md) for the layout and the
appearance choices.

| Video | Content |
| --- | --- |
| [`dose_sweep_DLDP_086.mp4`](docs/videos/dose_sweep_DLDP_086.mp4) | Planned versus predicted dose, sweeping caudal to cranial. Largest target of the 20 test cases (278 cc) and the most organs at risk in play (7 above 20 Gy) — also the worst dose score |
| [`dose_sweep_DLDP_083.mp4`](docs/videos/dose_sweep_DLDP_083.mp4) | The same, for the next-largest target (235 cc) |
| [`dose_sweep_DLDP_097.mp4`](docs/videos/dose_sweep_DLDP_097.mp4) | The same, for the demanding case whose fine streak structure the model reproduces best (216 cc, 6 OARs above 20 Gy) |
| [`optic_nerve_sensitivity.mp4`](docs/videos/optic_nerve_sensitivity.mp4) | Ten plausible left optic nerve contours: the re-optimised plan beside the prediction, with the mean dose each delivers to the contour |
| [`robustness_concave.mp4`](docs/videos/robustness_concave.mp4) | A concave target: planned dose, initial model, model retrained with concave cases |
| [`robustness_multiple.mp4`](docs/videos/robustness_multiple.mp4) | A multi-lesion target, same three panels |

Each frame puts the planning CT in grey under a dose heat map, outlines the target
volume and the organs at risk, and carries a structure legend and a dose colour bar.
Cases are chosen by measured properties rather than by eye — target size, how many
organs at risk the plan has to work around, and how well the prediction reproduces
the plan's fine streak structure — and the selector prints the full ranking of all
20 test cases before picking. The retrained-model clips additionally skip any case
the model was retrained on, which the scores give away.

Rebuild them with:

```bash
scripts/fetch_artifacts.sh
scripts/make_videos.sh
```

---

## Model

A two-level cascaded 3D U-Net ([Liu et al., Med Phys 2021](https://doi.org/10.1002/mp.15034)),
the architecture that won the [OpenKBP challenge](https://doi.org/10.1002/mp.14845).
The second U-Net takes the first one's output concatenated with its input.

- **Input:** 15 channels at 128³ — the normalised planning CT, the target volume and
  13 organ-at-risk masks.
- **Output:** a continuous dose distribution, scaled to 0–70 Gy.
- **Training:** 60 cases, 15 validation, 80 000 iterations, `0.5·L1(ref, coarse) + L1(ref, refined)`.
- **Inference:** four-flip test-time augmentation. About 45 s per case on Apple
  silicon (MPS), 15 s on an A5000 GPU.

Prescription was 60 Gy in 30 fractions, normalised so 100% of the dose covers 50%
of the target volume.

---

## Results, as reproduced from the archived artifacts

`scripts/reproduce.sh` recomputes every published number from the NIfTI volumes and
prints it beside the paper's value. Headlines:

| Quantity | Paper | Reproduced |
| --- | --- | --- |
| Dose score, 20 test cases (Cancers) | 0.94 (0.36) Gy | **0.94 (0.36) Gy** |
| DVH score, 20 test cases (Cancers) | 1.95 Gy | **1.96 Gy** |
| Per-OAR dose and DVH scores (ISBI Table 1, 26 values) | — | **all match to ≤ 0.002 Gy** |
| Optic nerve sensitivity (ISBI Table 2, 27 values) | — | **all match to ≤ 0.001** |
| Correlation, predicted vs planned dose shift | 0.926 | **0.926** |
| Correlation, planned dose shift vs Dice | −0.471 | **−0.471** |
| Cancers Table 3 (48 values) | — | **47 match to ≤ 0.01 Gy** |

Inference from the archived weights regenerates the archived predictions to
3 × 10⁻⁴ Gy on a 0–70 Gy scale (`python -m deepdosesens.analyze.verify_inference`), so
the tables above are checked against the model, not only against saved predictions.

The two papers report the overall dose score over slightly different scoring regions;
the value reproduced here, 0.94 Gy, is the one the journal version reports.

---

## Getting started

### Requirements

- Python 3.9+
- PyTorch (CUDA, MPS or CPU)
- SimpleITK, NumPy, SciPy, pandas, matplotlib
- ffmpeg, for the videos

```bash
git clone https://github.com/amithjkamath/deepdosesens.git
cd deepdosesens
uv venv .venv
source .venv/bin/activate
uv pip install -r pyproject.toml
```

### Configuring where the data lives

No paths are hardcoded. Copy `.env.example` to `.env` and point it at your copy of
the artifacts (or set the same variables in the environment):

```ini
DEEPDOSESENS_ISBI_ARCHIVE=/path/to/2022-11-ISBI/artifacts
DEEPDOSESENS_CANCERS_ARCHIVE=/path/to/2023-08-Cancers/artifacts
#DEEPDOSESENS_DATA=/mnt/big-disk/deepdosesens/data
#DEEPDOSESENS_CHECKPOINTS=/mnt/big-disk/deepdosesens/checkpoints
```

Everything falls back to directories inside the repository, so the defaults work
once `scripts/fetch_artifacts.sh` has unpacked the archives. Check what is in
effect with:

```bash
python -m deepdosesens.config
```

Only code and the demonstration videos are committed here. Planning CTs, contours,
reference plans, predicted dose volumes and model weights live in the artifact
archives and are shared on request; each archive carries its own manifest describing
the layout. Fetch from it with:

```bash
WHAT=isbi scripts/fetch_artifacts.sh   # ISBI data, predictions and weights
scripts/fetch_artifacts.sh             # adds the Cancers cases and models
```

### Predicting dose

The model is built and its weights read once; predicting a test set is one load and
N forward passes:

```python
from deepdosesens.config import checkpoint_path, data_path
from deepdosesens.inference import DosePredictor

predictor = DosePredictor(checkpoint_path("dose-predictor", "weights.pt"))
for case in ["DLDP_081", "DLDP_082"]:
    dose = predictor.predict(data_path("glioblastoma", case))  # Gy, (Z, H, W)
```

or from the command line:

```bash
python -m deepdosesens.inference --run my-run          # the 20 test cases
python -m deepdosesens.analyze.verify_inference        # check against the archive
```

### Training

```bash
python train_C3D.py --batch_size 2 --max_iter 80000    # the cascaded 3D U-Net
python train_UNet.py                                   # single U-Net baseline
```

---

## Repository layout

| Path | Contents |
| --- | --- |
| `deepdosesens/config.py` | every path the project uses, from env vars or `.env` |
| `deepdosesens/inference.py` | `DosePredictor` — load once, predict many |
| `deepdosesens/model/` | the cascaded 3D U-Net, loss, and checkpoint loading/slimming |
| `deepdosesens/data/` | reading, preprocessing and augmentation |
| `deepdosesens/analyze/` | score definitions, the two reproduction scripts and the inference check |
| `deepdosesens/visualization/` | the video builders and their shared drawing code |
| `scripts/` | `fetch_artifacts.sh`, `reproduce.sh`, `make_videos.sh` |
| `examples/` | notebooks from the original analysis |

If this is useful in your research, please consider citing:

    @article{poel2023deep,
      title={Deep-Learning-Based Dose Predictor for Glioblastoma--Assessing the Sensitivity and Robustness for Dose Awareness in Contouring},
      author={Poel, Robert and Kamath, Amith J and Willmann, Jonas and Andratschke, Nicolaus and Ermi{\c{s}}, Ekin and Aebersold, Daniel M and Manser, Peter and Reyes, Mauricio},
      journal={Cancers},
      volume={15},
      number={17},
      pages={4226},
      year={2023}
    }

    @inproceedings{kamath2023doseprediction,
      title={How sensitive are deep learning based radiotherapy dose prediction models to variability in Organs At Risk segmentation?},
      author={Kamath, Amith and Poel, Robert and Willmann, Jonas and Andratschke, Nicolaus and Reyes, Mauricio},
      booktitle={2023 IEEE 20th International Symposium on Biomedical Imaging (ISBI)},
      pages={1--4},
      year={2023},
      organization={IEEE}
    }

## Credits

Major props to the code and organization in
https://github.com/LSL000UD/RTDosePrediction, which is what this model is based on
(looks like this repo is not maintained/available anymore!)
