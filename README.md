Deep Learning Dose Prediction Sensitivity for Glioblastoma
==============================

This repository accompanies our paper: "How Sensitive Are Deep Learning Based Radiotherapy Dose Prediction Models To Variability In Organs At Risk Segmentation?" accepted at the 20th International Symposium for Biomedical Imaging, 2023. 

If this is useful in your research, please consider citing:

    @inproceedings{kamath2023doseprediction,
    title={How sensitive are deep learning based radiotherapy dose prediction models to variability in Organs At Risk segmentation?},
    author={Kamath, Amith and Poel, Robert and Willmann, Jonas and Andratschke, Nicolaus and Reyes, Mauricio},
    booktitle={2023 IEEE 20th International Symposium on Biomedical Imaging (ISBI)},
    pages={1--4},
    year={2023},
    organization={IEEE}
    }

Other instructions are TBD.

Project Organization
------------

    ├── data
    │   └── .gitkeep            <- empty in this repo; contact us for access.
    │
    ├── models                  
    |   └── .gitkeep            <- empty at this point; contact us for access.
    │
    ├── deepdosesens            <- Source code for use in this project.
    │   │
    │   ├── data                    <- Scripts to download or generate data
    │   │
    │   ├── model                   <- Classes to define the model architecture and losses.
    │   │
    │   ├── training                <- Classes to handle network training.
    │   │
    │   ├── utils                   <- Scripts utilities used during data generation or training
    │   │
    │   ├── validation              <- Classes to evaluate model performance.
    │   │
    │   ├── __init__.py             <- package initialization.
    │   ├── test_per_ROI_scores.py  <- Generate dose and DVH scores per ROI.
    │   ├── test.py                 <- model testing script.
    │   └── train.py                <- Training script.
    │
    ├── docs                    <- A default Sphinx project; TBD for details.
    |
    ├── notebooks               <- Jupyter notebooks. 
    │   ├── 1.0-ajk-visual-dose-differences.ipynb           <- Generate figure 2.
    │   ├── 2.0-ajk-per-OAR-table.ipynb                     <- Generate table 1.
    │   ├── 3.0-ajk-generate-ONL-DVHs.ipynb                 <- Generate figure 3.
    │   ├── 4.0-ajk-generate-ONL-sensitivity-table.ipynb    <- Generate table 2.
    │   └── 5.0-ajk-generate-ONL-differences.ipynb          <- additional figures, unused in paper.
    |
    ├── results                 <- Where results from the notebooks are stored.    
    |
    ├── .gitignore              <- .gitignore for this project.
    ├── LICENSE                 <- LICENSE for this project.
    ├── README.md               <- The top-level README for developers using this project.
    ├── requirements.txt        <- The requirements file for reproducing the analysis     
    │                              environment, generated with `pip freeze > 
    │                              requirements.txt`
    └── setup.py                <- boilerplate for using with pip.


Credits
------------
Major props to the code and organization in https://github.com/LSL000UD/RTDosePrediction, which is what this model is based on.