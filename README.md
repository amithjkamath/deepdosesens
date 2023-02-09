deep-planner
==============================

This repository explores the Deep Planner project.

To create a new test set, run the following scripts (in the deep-planner/utils/ folder) in this specific order:
1. rtss_to_nifti.py
2. convert_dose_volume.py
3. resize_to_standard_dimensions.py

Then train the model using the train.py script (modifying the paths to the data files of course).

and finally, test the model performance using test.py

Project Organization
------------

    ├── LICENSE
    ├── README.md                               <- The top-level README for developers using this project.
    ├── data
    │   ├── processed                           <- The final, canonical data sets for modeling.
    │   └── raw                                 <- The original, immutable data dump.
    │
    ├── docs                                    <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models                                  <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks                               <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                                              the creator's initials, and a short `-` delimited description, e.g.
    │                                              `1.0-jqp-initial-data-exploration`.
    │
    ├── references                              <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports                                 <- Generated analysis as HTML, PDF, LaTeX, etc.
    │
    ├── requirements.txt                        <- The requirements file for reproducing the analysis environment, e.g.
    │                                              generated with `pip freeze > requirements.txt`
    │
    ├── deep-planner           <- Source code for use in this project.
    │   │
    │   ├── data                                <- Scripts to download or generate data
    │   │
    │   ├── model                               <- Classes to define the model architecture and losses.
    │   │
    │   ├── utils                               <- Scripts utilities used during data generation or training
    │   │
    │   ├── training                            <- Classes to handle network training.
    │   │
    │   ├── validation                          <- Classes to evaluate model performance.
    │   │
    │   └── visualization                       <- Scripts to create exploratory and results oriented visualizations
