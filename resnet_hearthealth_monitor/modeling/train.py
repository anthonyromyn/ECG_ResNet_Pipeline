from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
import os
import gc
import joblib
import shutil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np
from sklearn.model_selection import StratifiedKFold

import optuna
import mlflow
import mlflow.pytorch
from mlflow.models.signature import infer_signature

from resnet_hearthealth_monitor.config import MODELS_DIR, BEST_MODEL_DIR
from resnet_hearthealth_monitor.data_processing.pytorch_datasets import augmentable_dataset
from resnet_hearthealth_monitor.data_processing.augmentations import Augment

app = typer.Typer()



def train_model(trial_dir, train_set, torch_model_class, torch_dataset_class, augmentations, n_splits, epochs, device, early_stopping_patience=6, **kwargs):
    # Lists to store the epoch-wise losses for each fold.
    all_fold_train_losses = []
    all_fold_val_losses = []
    best_val_losses_per_fold = []
    best_epochs_per_fold = []

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    augment = Augment(augmentations, False)
    lr = kwargs.get('lr')
    batch_size = kwargs.get('batch_size')
    if kwargs.get('weight_decay') is not None:
        weight_decay = kwargs['weight_decay']
    else:
        weight_decay = 0.01

    if augment.return_prints == True:
        logger.info("augment.return_prints == True in train_model(), this will lead to a crash at dataset creation, change to False.")
    for fold_n, (train_idx, val_idx) in enumerate(
        skf.split(train_set.iloc[:,:-1], train_set.iloc[:,-1])
    ):
        # Create datasets for this fold
        train_subset = torch_dataset_class(train_set.iloc[train_idx, :-1], train_set.iloc[train_idx, -1], transforms=augment)

        val_subset = torch_dataset_class(train_set.iloc[val_idx, :-1], train_set.iloc[val_idx, -1])
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size * 4)
        
        # Initialize model, optimizer, scheduler, and loss criterion
        model = torch_model_class.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        lr_sched = ReduceLROnPlateau(optimizer, patience=3)
        weights = torch.tensor([0.2415, 7.875, 3.026, 27.31, 2.723], dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)  # Adjust if necessary
        
        # To store losses for each epoch in this fold
        epoch_train_losses = []
        epoch_val_losses = []
        
        # Initialize early stopping variables for this fold
        best_val_loss_fold = float('inf')
        best_epoch_fold = 0
        epochs_without_improvement = 0

        for epoch in range(epochs):
            # --- Training phase ---
            model.train()
            running_train_loss = 0.0
            for x, y_true in train_loader:
                optimizer.zero_grad()
                y_pred = model(x.to(device))
                loss = criterion(y_pred, y_true.long().to(device))
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item() / len(train_loader)
            epoch_train_loss = running_train_loss
            
            # --- Validation phase ---
            model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for x, y_true in val_loader:
                    y_pred = model(x.to(device))
                    loss = criterion(y_pred, y_true.long().to(device))
                    running_val_loss += loss.item() / len(val_loader)
            epoch_val_loss = running_val_loss
            
            # Step the learning rate scheduler based on the validation loss.
            lr_sched.step(epoch_val_loss)
            
            # Save the losses for this epoch
            epoch_train_losses.append(epoch_train_loss)
            epoch_val_losses.append(epoch_val_loss)
            print(f'Fold {fold_n}, Epoch {epoch}: Train loss {epoch_train_loss:.2e}, Val loss {epoch_val_loss:.2e}, LR {optimizer.param_groups[0]["lr"]:.5e}')

            # Early stopping: check if validation loss improved
            if epoch_val_loss < best_val_loss_fold:
                best_val_loss_fold = epoch_val_loss
                best_epoch_fold = epoch
                epochs_without_improvement = 0  # reset counter
                # Save the model state dict
                save_path = os.path.join(trial_dir, f"model_fold_{fold_n}_best.pth")
                torch.save(model.state_dict(), save_path)
            else:
                epochs_without_improvement += 1
            
            # If no improvement for early_stopping_patience epochs, break
            if epochs_without_improvement >= early_stopping_patience:
                print(f'Fold {fold_n}: Early stopping triggered at epoch {epoch}')
                break

        # Save best loss and epoch for this fold
        best_val_losses_per_fold.append(best_val_loss_fold)
        best_epochs_per_fold.append(best_epoch_fold)
        all_fold_train_losses.append(epoch_train_losses)
        all_fold_val_losses.append(epoch_val_losses)

    # Average losses
    avg_best_val_loss = np.mean(best_val_losses_per_fold)
    avg_best_val_loss_epoch = np.mean(best_epochs_per_fold)

    print(f'Average best validation loss: {avg_best_val_loss:.2e} at average epoch {avg_best_val_loss_epoch:.1f}')
    
    # Return overall average validation loss plus the epoch-wise curves for plotting.
    return avg_best_val_loss, avg_best_val_loss_epoch, best_val_losses_per_fold, best_epochs_per_fold, all_fold_train_losses, all_fold_val_losses

# A helper function to parse hyperparameter_space dynamically.
def suggest_params(trial, hp_space):
    params = {}
    for key, spec in hp_space.items():
        if spec['type'] == 'float':
            params[key] = trial.suggest_float(key, spec['low'], spec['high'], log=spec.get('log', False))
        elif spec['type'] == 'categorical':
            params[key] = trial.suggest_categorical(key, spec['choices'])
        elif spec['type'] == 'int':
            params[key] = trial.suggest_int(key, spec['low'], spec['high'], log=spec.get('log', False))
        else:
            raise ValueError(f"Unsupported hyperparameter type: {spec['type']} for {key}")
    return params

def optuna_objective(trial, train_set, torch_model_class, torch_dataset_class, augmentations, hyperparameter_space, n_epochs, n_splits, device):
    # First, make sure caches are cleared of unneeded data
    gc.collect()
    torch.cuda.empty_cache()

    # Start a nested MLflow run for this trial
    with mlflow.start_run(nested=True):
        # Create a folder for the study (if it doesn't already exist)
        study_folder = f"../models/{trial.study.study_name}"
        os.makedirs(study_folder, exist_ok=True)
        # Create a subfolder for this trial within the study folder
        trial_dir = os.path.join(study_folder, f"trial_{trial.number}")
        os.makedirs(trial_dir, exist_ok=True)

        # Hyperparameters for this trial
        params = suggest_params(trial, hyperparameter_space)
        print(params)
        # Log hyperparameters in MLflow
        mlflow.log_params(params)
        mlflow.log_param("epochs", n_epochs)
        mlflow.log_param("n_splits", n_splits)

        avg_best_val_loss, avg_best_val_loss_epoch, best_val_losses_per_fold, best_epochs_per_fold, \
            all_fold_train_losses, all_fold_val_losses = train_model(
            trial_dir, train_set, torch_model_class, torch_dataset_class, augmentations, n_splits, n_epochs, 
            device, early_stopping_patience=6, **params
        )

        # Log metrics for this trial in MLflow
        mlflow.log_metric("avg_best_val_loss", avg_best_val_loss)

        # Save model paths as trial attributes
        trial.set_user_attr("model_paths", {fold_n: os.path.join(trial_dir, f"model_fold_{fold_n}_best.pth")
                                        for fold_n in range(n_splits)})
        
        # Additionally, save them to MLFlow
        dummy_input_tensor = torch.rand(1,1,125).to(device)
        with torch.no_grad():
                model = torch_model_class.to(device)
                dummy_output_tensor = model(dummy_input_tensor)
        dummy_input = dummy_input_tensor.cpu().numpy()
        dummy_output = dummy_output_tensor.cpu().numpy()   
        signature = infer_signature(dummy_input, dummy_output)

        for fold_n, model_path in trial.user_attrs["model_paths"].items():
            model = torch_model_class.to(device)
            checkpoint_path = model_path
            model.load_state_dict(torch.load(checkpoint_path))
            mlflow.pytorch.log_model(model, f"model_fold_{fold_n}_best", signature=signature)
        
        trial.set_user_attr("avg_best_val_loss", avg_best_val_loss)  
        trial.set_user_attr("avg_best_val_loss_epoch", avg_best_val_loss_epoch)
        trial.set_user_attr("best_val_losses_per_fold", best_val_losses_per_fold)
        trial.set_user_attr("best_epochs_per_fold", best_epochs_per_fold)
        trial.set_user_attr("all_fold_train_losses", all_fold_train_losses)
        trial.set_user_attr("all_fold_val_losses", all_fold_val_losses)

        return avg_best_val_loss
    

@app.command(name="run_study")
def run_study(study_name, n_trials, train_set, torch_model_class, torch_dataset_class, augmentations, hyperparameter_space, n_epochs, n_splits, device):
    # setup MLFlow and Optuna
    mlruns_dir = os.path.join("../models/mlruns")
    os.makedirs(mlruns_dir, exist_ok=True)
    mlflow.set_tracking_uri("file://" + os.path.abspath(mlruns_dir))
    mlflow.set_experiment(study_name)
    study = optuna.create_study(direction='minimize', study_name=study_name)
    logger.success(f"Optuna study and MLFlow records initiated in {MODELS_DIR}")

    if hyperparameter_space==None:
        hyperparameter_space = {
            'lr': {
                'type': 'float',
                'low': 1e-5,
                'high': 1e-2,
                'log': True
            },
            'batch_size': {
                'type': 'categorical',
                'choices': [32, 64, 128, 256, 512]
            },
            'weight_decay': {
                'type': 'float',
                'low': 1e-5,
                'high': 1e-1,
                'log': True
            },
        }
    study.optimize(lambda trial: optuna_objective(trial, train_set, torch_model_class, torch_dataset_class, augmentations, hyperparameter_space, n_epochs, n_splits, device), n_trials)

    logger.success(f"Best hyperparameters: {study.best_params}")
    logger.success(f"Best validation loss: {study.best_value}")

    # Save the entire study object (all trials and results) to the ../models directory.
    import joblib
    models_dir = os.path.join("..", "models")
    os.makedirs(models_dir, exist_ok=True)
    study_filename = os.path.join(models_dir, f"{study.study_name}.pkl")
    joblib.dump(study, study_filename)
    logger.success(f"Study saved to {MODELS_DIR} under filename {study.study_name}")


@app.command(name="find_best_study")
def find_best_study():
    logger.info(f"Loading all model searches stored in {MODELS_DIR}.")
    # List all .pkl files in the models_dir
    pkl_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pkl")]

    best_score = None
    best_study_file = None

    logger.info(f"Examining best performances from each search.")    
    # Loop over each .pkl file to find the one with the lowest best_trial.value
    for pkl_file in pkl_files:
        file_path = os.path.join(MODELS_DIR, pkl_file)
        try:
            study = joblib.load(file_path)
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")
            continue

        # Access the best trial and its value
        best_trial = study.best_trial
        trial_value = best_trial.value  # assuming lower is better
        print(f"Study file '{pkl_file}' has best_trial.value: {trial_value}")

        if best_score is None or trial_value < best_score:
            best_score = trial_value
            best_study_file = file_path

    if best_study_file is not None:
        logger.success(f"Best study file is: '{best_study_file}' with score {best_score}")

        # Ensure the BEST_MODEL_DIR exists; if not, create it.
        os.makedirs(BEST_MODEL_DIR, exist_ok=True)

        # Remove any existing .pkl files in BEST_MODEL_DIR
        for f in os.listdir(BEST_MODEL_DIR):
            if f.endswith(".pkl"):
                os.remove(os.path.join(BEST_MODEL_DIR, f))
                print(f"Removed existing file: {f}")

        # Copy the best study file into the BEST_MODEL_DIR
        try:
            shutil.copy(best_study_file, BEST_MODEL_DIR)
            logger.success(f"Copied '{best_study_file}' to '{BEST_MODEL_DIR}'")
        except Exception as e:
            logger.info(f"Failed to copy '{best_study_file}' to '{BEST_MODEL_DIR}': {e}")
    else:
        logger.info("No valid study files found.")

def load_best_study():
    """
    Load the best Optuna study from BEST_MODEL_DIR.
    
    This function searches the directory defined by BEST_MODEL_DIR for any files with a 
    .pkl extension (which are expected to be serialized Optuna studies). It then loads the 
    first .pkl file found using joblib and returns this study object.
    
    Raises:
        FileNotFoundError: If no .pkl files are found in the BEST_MODEL_DIR.
    
    Returns:
        best_study: Loads the Optuna study in BEST_MODEL_DIR
    """
    best_model_study_dir = BEST_MODEL_DIR
    pkl_files = [f for f in os.listdir(best_model_study_dir) if f.endswith(".pkl")]

    if not pkl_files:
        raise FileNotFoundError("No .pkl files found in the best_model folder.")

    best_model_study_file = pkl_files[0]
    best_model_study_path = os.path.join(best_model_study_dir, best_model_study_file)
    best_optuna_study = joblib.load(best_model_study_path)

    return best_optuna_study

def load_best_model():
    """
    Load the best Optuna study from BEST_MODEL_DIR and retrieve its best trial.
    
    This function searches the directory defined by BEST_MODEL_DIR for any files with a 
    .pkl extension (which are expected to be serialized Optuna studies). It then loads the 
    first .pkl file found using joblib, retrieves the best trial (e.g., the trial with the 
    lowest validation loss if the study is set to 'minimize'), and returns this trial object.
    
    Raises:
        FileNotFoundError: If no .pkl files are found in the BEST_MODEL_DIR.
    
    Returns:
        best_trial: The best trial object from the loaded Optuna study.
    """
    best_model_study_dir = BEST_MODEL_DIR
    pkl_files = [f for f in os.listdir(best_model_study_dir) if f.endswith(".pkl")]

    if not pkl_files:
        raise FileNotFoundError("No .pkl files found in the best_model folder.")

    best_model_study_file = pkl_files[0]
    best_model_study_path = os.path.join(best_model_study_dir, best_model_study_file)
    best_optuna_study = joblib.load(best_model_study_path)
    # Retrieve the best trial
    best_trial = best_optuna_study.best_trial
    return best_trial

if __name__ == "__main__":
    app()
