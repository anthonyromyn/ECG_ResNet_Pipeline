from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
import torch
import os
import joblib
from statistics import mode

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from resnet_hearthealth_monitor.config import BEST_MODEL_DIR, FIGURES_DIR
from resnet_hearthealth_monitor.modeling.train import load_best_model
from resnet_hearthealth_monitor.plots import save_classif_figure

app = typer.Typer()

def testset_prediction(test_set, torch_model_class, torch_dataset_class, device, save_and_display_results):
    best_model = load_best_model()

    test_set = torch_dataset_class(test_set.iloc[:,:-1], test_set.iloc[:,-1])
    batch_size=best_model.params['batch_size']
    test_loader = DataLoader(test_set, batch_size=batch_size*4, shuffle=False, drop_last=False)

    predictions = []

    n_splits = len(best_model.user_attrs['best_epochs_per_fold'])
    with torch.no_grad():
        for fold in range(n_splits):
            model = torch_model_class.to(device)
            checkpoint_path = best_model.user_attrs['model_paths'][fold]
            model.load_state_dict(torch.load(checkpoint_path))
            model.eval()
            
            fold_predictions = []
            labels = []

            for batch_id, (x,y_true) in enumerate(test_loader):
                y_pred = model(x.to(device)).argmax(1)

                fold_predictions.extend(y_pred.squeeze().tolist())
                labels.extend(y_true.squeeze().tolist())
            predictions.append(fold_predictions)

    predictions_and_labels = pd.DataFrame(
        np.array(predictions).T, 
        columns=[f'fold {i}' for i in range(n_splits)]
    )
    predictions_and_labels['Label'] = labels

    final_predictions = predictions_and_labels.iloc[:, :n_splits].mode(axis=1)[0]

    final_predictions_and_labels = pd.DataFrame({
        'Prediction': final_predictions,
        'Label': predictions_and_labels['Label']
    })

    save_path = os.path.join(BEST_MODEL_DIR, "testset_predictions_and_labels.csv")
    if save_and_display_results == True:
        final_predictions_and_labels.to_csv(save_path, index=False)
        logger.success(f'Test-set predictions saved to {BEST_MODEL_DIR}/testset_predictions_and_labels.csv.')
        print("True label distribution:")
        print(pd.Series(final_predictions_and_labels['Label']).value_counts())
        print("\nClassification Report:")
        print(classification_report(final_predictions_and_labels['Label'], final_predictions_and_labels['Prediction']))
        save_path = f"{FIGURES_DIR}/classification_report.png"
        print("final_predictions_and_labels['Label']:")
        print(final_predictions_and_labels['Label'])
        print("final_predictions_and_labels['Prediction']:")
        print(final_predictions_and_labels['Prediction'])
        print("save_path:")
        print(save_path)
        save_classif_figure(final_predictions_and_labels['Label'], final_predictions_and_labels['Prediction'], save_path)
        logger.success(f'classification report saved to {FIGURES_DIR}/classification_report.png.')
    else:
        logger.success('test-set predictions generated')
    return final_predictions_and_labels

# @app.command(name="testset_prediction")
# def testset_prediction(
#     test_set_path: str = typer.Argument(
#         "data/raw/mitbih_test.csv",
#         help="Path to the test CSV file containing features and labels."
#     ),
#     torch_model_class_path: str = typer.Argument(
#         "resnet_hearthealth_monitor.modeling.resnet.ResNet125",
#         help="Module path for the PyTorch model class, e.g., 'resnet_hearthealth_monitor.modeling.resnet.ResNet125'."
#     ),
#     torch_dataset_class_path: str = typer.Argument(
#         "resnet_hearthealth_monitor.data_processing.pytorch_datasets.augmentable_dataset",
#         help="Module path for the PyTorch dataset class, e.g., 'resnet_hearthealth_monitor.data_processing.pytorch_datasets.augmentable_dataset'."
#     ),
#     device: str = typer.Argument(
#         "cuda" if torch.cuda.is_available() else "cpu",
#         help="Device to run inference on (e.g., 'cuda' or 'cpu')."
#     ),
#     save_and_display_results: bool = typer.Option(
#         True,
#         "--save-results",
#         help="If True, save predictions to CSV and display the classification report."
#     )
# ):
#     """
#     Generate test-set predictions using the best model from a previous study.

#     This command loads the best model (retrieved from BEST_MODEL_DIR), constructs a dataset
#     from the provided test CSV file, runs inference, and either displays and saves the results or
#     simply returns them.
#     """
#     # Dynamic import for torch_model_class
#     model_module_path, model_class_name = torch_model_class_path.rsplit(".", 1)
#     torch_model_module = __import__(model_module_path, fromlist=[model_class_name])
#     torch_model_class = getattr(torch_model_module, model_class_name)
    
#     # Similarly load dataset class (if needed)
#     dataset_module_path, dataset_class_name = torch_dataset_class_path.rsplit(".", 1)
#     torch_dataset_module = __import__(dataset_module_path, fromlist=[dataset_class_name])
#     torch_dataset_class = getattr(torch_dataset_module, dataset_class_name)

#     # Load the test set from CSV.
#     test_set = pd.read_csv(test_set_path, header=None)
#     # Create the dataset instance.
#     test_set = torch_dataset_class(test_set.iloc[:, :-1], test_set.iloc[:, -1])
    
#     # Retrieve the best model from the study.
#     best_model = load_best_model()
#     batch_size = best_model.params['batch_size']
#     test_loader = DataLoader(test_set, batch_size=batch_size * 4, shuffle=False, drop_last=False)

#     predictions = []
#     n_splits = len(best_model.user_attrs['best_epochs_per_fold'])
#     with torch.no_grad():
#         for fold in range(n_splits):
#             model = torch_model_class().to(device)
#             checkpoint_path = best_model.user_attrs['model_paths'][fold]
#             model.load_state_dict(torch.load(checkpoint_path))
#             model.eval()

#             fold_predictions = []
#             labels = []
#             for _, (x, y_true) in enumerate(test_loader):
#                 y_pred = model(x.to(device)).argmax(dim=1)
#                 fold_predictions.extend(y_pred.squeeze().tolist())
#                 labels.extend(y_true.squeeze().tolist())
#             predictions.append(fold_predictions)

#     predictions_and_labels = pd.DataFrame(
#         np.array(predictions).T,
#         columns=[f'fold {i}' for i in range(n_splits)]
#     )
#     predictions_and_labels['Label'] = labels
#     final_predictions = predictions_and_labels.iloc[:, :n_splits].mode(axis=1)[0]
#     final_predictions_and_labels = pd.DataFrame({
#         'Prediction': final_predictions,
#         'Label': predictions_and_labels['Label']
#     })

#     save_path = os.path.join(BEST_MODEL_DIR, "testset_predictions_and_labels.csv")
#     if save_and_display_results:
#         final_predictions_and_labels.to_csv(save_path, index=False)
#         typer.echo(f"Test-set predictions saved to {save_path}.")
#         typer.echo("True label distribution:")
#         typer.echo(pd.Series(final_predictions_and_labels['Label']).value_counts().to_string())
#         typer.echo("\nClassification Report:")
#         typer.echo(classification_report(final_predictions_and_labels['Label'], final_predictions_and_labels['Prediction']))
#         fig_path = os.path.join(FIGURES_DIR, "classification_report.png")
#         save_classif_figure(final_predictions_and_labels['Label'], final_predictions_and_labels['Prediction'], fig_path)
#         typer.echo(f"Classification report saved to {fig_path}.")
#     else:
#         typer.echo("Test-set predictions generated.")

#     return final_predictions_and_labels

if __name__ == "__main__":
    app()
