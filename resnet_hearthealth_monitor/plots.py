from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import classification_report
import pandas as pd

from resnet_hearthealth_monitor.config import FIGURES_DIR, BEST_MODEL_DIR
from resnet_hearthealth_monitor.modeling.train import load_best_model

app = typer.Typer()

@app.command(name="gen_training_loss_plots")
def gen_training_loss_plots():
    best_model = load_best_model()
    # Plot 1: Averaged learning curve across folds with padding
    all_fold_train_losses = best_model.user_attrs['all_fold_train_losses']
    all_fold_val_losses = best_model.user_attrs['all_fold_val_losses']
    
    # Find maximum number of epochs among folds
    max_epochs = max(len(losses) for losses in all_fold_train_losses)

    # Pad each fold's loss list with its last value so that all have length max_epochs
    padded_train_losses = []
    padded_val_losses = []

    for losses in all_fold_train_losses:
        pad_length = max_epochs - len(losses)
        padded = losses + [losses[-1]] * pad_length
        padded_train_losses.append(padded)

    for losses in all_fold_val_losses:
        pad_length = max_epochs - len(losses)
        padded = losses + [losses[-1]] * pad_length
        padded_val_losses.append(padded)

    # Compute average loss across folds for each epoch
    avg_train_losses = np.mean(padded_train_losses, axis=0)
    avg_val_losses = np.mean(padded_val_losses, axis=0)

    # Plot the averaged curves
    epochs = np.arange(max_epochs)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, avg_train_losses, label='Average Train Loss', linestyle='--')
    plt.plot(epochs, avg_val_losses, label='Average Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Average Learning Curve Across Folds (with padding)')
    plt.legend()
    
    # Save the figure
    plot1_path = os.path.join(FIGURES_DIR, "avg_learning_curve_over_folds.png")
    plt.savefig(plot1_path)
    print(f"Saved average learning curve plot to {plot1_path}")
    plt.show()

    # Plot 2: Learning curves per fold
    colors = plt.get_cmap('tab10').colors

    plt.figure(figsize=(10, 6))
    for i, (train_losses, val_losses) in enumerate(zip(all_fold_train_losses, all_fold_val_losses)):
        epochs_fold = np.arange(len(train_losses))
        color = colors[i % len(colors)]  # Cycle through colors if more folds than available colors
        plt.plot(epochs_fold, train_losses, label=f'Fold {i} Train', color=color, alpha=0.8, linestyle='-')
        plt.plot(epochs_fold, val_losses, label=f'Fold {i} Val', color=color, alpha=0.8, linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curves per Fold')
    plt.legend()

    # Save the second figure
    plot2_path = os.path.join(FIGURES_DIR, "learning_curves_per_fold.png")
    plt.savefig(plot2_path)
    print(f"Saved fold learning curves plot to {plot2_path}")
    plt.show()

    logger.success(f"Loss plots generated and saved to {FIGURES_DIR}")

@app.command(name="save_classif_figure")
def save_classif_figure(
    true_labels=None,
    predicted_labels=None,
    save_path: Path = None
):
    """
    Generate a classification report plot from labels or CSV paths.

    If called from CLI without arguments, loads defaults from BEST_MODEL_DIR.
    """
    # Handle CLI default behavior
    if true_labels is None or predicted_labels is None or save_path is None:
        csv_path = os.path.join(BEST_MODEL_DIR, "testset_predictions_and_labels.csv")
        df = pd.read_csv(csv_path)
        true_labels = df["Label"]
        predicted_labels = df["Prediction"]
        save_path = os.path.join(FIGURES_DIR, "classification_report.png")

    # If given as string/Path (from CLI), load from file
    elif isinstance(true_labels, (str, Path)) and isinstance(predicted_labels, (str, Path)):
        true_labels = pd.read_csv(true_labels, header=None).squeeze()
        predicted_labels = pd.read_csv(predicted_labels, header=None).squeeze()

    # Otherwise we assume they are passed as arrays/pd.Series directly
    report = classification_report(true_labels, predicted_labels)

    plt.figure(figsize=(10, 6))
    plt.text(0.01, 0.05, report, {'fontsize': 12}, fontproperties='monospace')
    plt.axis('off')
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(FIGURES_DIR, "classification_report.png")
    plt.savefig(save_path)
    plt.show()
    logger.success(f"Classification report figure saved to: {save_path}")

if __name__ == "__main__":
    app()
