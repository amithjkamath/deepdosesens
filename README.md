deepdosesens
==============================

This repository accompanies our paper: "How Sensitive Are Deep Learning Based Radiotherapy Dose Prediction Models To Variability In Organs At Risk Segmentation?" accepted at the 20th International Symposium for Biomedical Imaging, 2023. 

Other instructions are TBD.

Project Organization
------------

    ├── LICENSE
    ├── README.md                   <- The top-level README for developers using this project.
    ├── data
    │   └── .gitkeep                <- Folder for the data set; empty in this repo; contact us for details.
    │
    ├── docs                        <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models                      <- Trained model; also empty at this point; contact us for details.
    │
    ├── notebooks                   <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                                  the creator's initials, and a short `-` delimited description, e.g.
    │                                  `1.0-jqp-initial-data-exploration`.
    │
    ├── results                     <- Where results from the notebooks are stored.
    │
    ├── requirements.txt            <- The requirements file for reproducing the analysis environment, e.g.
    │                                  generated with `pip freeze > requirements.txt`
    │
    └── deepdosesens            <- Source code for use in this project.
        │
        ├── data                <- Scripts to download or generate data
        │
        ├── model               <- Classes to define the model architecture and losses.
        │
        ├── utils               <- Scripts utilities used during data generation or training
        │
        ├── training            <- Classes to handle network training.
        │
        └── validation          <- Classes to evaluate model performance.
