# ResNet HeartHealth Monitor

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

# Primary Features
- accurate classification of five different heartbeat patterns, discerning healthy beats and 4 types of arrhythmias
- 2.6% increased performance with 0% bigger network over foundational model (96% vs 93.4% average accuracy)
- Analysis walk-through notebooks
- Optuna and MLFlow model tracking
- FastAPI-based model inference
- Make commands for easy setup 
- Cookie Cutter Data Science formatted for easy navigation

# To start using this repo:

To run the notebooks you must first please: 
1. run ```make create_environment```
2. run ```make data``` to generate the raw data
after which you can progress through the notebooks to see everything this repository has to offer.

## Additional Details

### Optuna
Optuna studies and saved models from searches are saved to ./models and ./models/best_model

### MLFlow
MLFlow records for these studies are saved to ./models under the Optuna study names and can be access with  ```mlflow ui```.

### FastAPI model inference
FastAPI-based model inference predictions server can be initated in ./resnet_hearthealth_monitor with command ```uvicorn inference_api:app --reload```

This API accepts a 1x125 list of float values (a single ECG sample) and provides a prediction from the model in ./models/best_model.

An example can be found in ./notebooks/chapter_4_deployment_strategies.ipynb

### Make commands
If interested, there are make commands available for a number of additional functions:
```
make preprocess                                     - Preprocessing the raw data
make find_best_study                                - Copy best model (and it's associated records) to ./models/best_model for prediction or plotting
make gen_training_loss_plots                        - generate training loss plots for best model and save to ./reports/figures
make save_classif_figure                            - generate classification report table for best model and save to ./reports/figures
make predict_segment                                - use best model to generate a prediction for a new sample
```

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│   ├── mlruns         <- MLFlow records
│   └── best_model     <- The best performing model is saved here
│
├── notebooks          <- Jupyter notebooks. 
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         resnet_hearthealth_monitor and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└──  resnet_hearthealth_monitor   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes resnet_hearthealth_monitor a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── data_processing 
    │   ├── __init__.py   
    │   ├── augmentations.py      <- Code for data augmentations and their application     
    │   ├── preprocess.py         <- Code to run preprocessing on raw data
    │   └── pytorch_datasets.py   <- Code for pytorch datasets used
    │
    ├── inference_api.py        <- Code for FastAPI-based inference capability
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── inference.py        <- Code used by inference API       
    │   ├── predict.py          <- Code to run model predictions with trained models   
    │   ├── resnet.py           <- Torch ResNet Model code
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

