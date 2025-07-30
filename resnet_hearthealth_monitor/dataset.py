from pathlib import Path
import typer
from loguru import logger
from tqdm import tqdm
import kagglehub
import shutil
import os
import pandas as pd

from resnet_hearthealth_monitor.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


def setup_raw_ecg_dataset():
    """Download ECG dataset from Kaggle if not already present..."""
    if (RAW_DATA_DIR / "mitbih_data.zip").exists():
        logger.info("ECG dataset found in the raw data directory as .zip file. Unzipping...")
        shutil.unpack_archive(RAW_DATA_DIR / "mitbih_data.zip", RAW_DATA_DIR)
        # After unzipping, move the data from the subfolder up one level to the RAW_DATA_DIR and remove the subfolder.
        subfolder = RAW_DATA_DIR / "mitbih_data"

        for item in subfolder.iterdir():
            destination = RAW_DATA_DIR / item.name
            if not destination.exists():
                shutil.move(str(item), str(RAW_DATA_DIR))
                logger.success(f"Moved {item} from extracted directory to {RAW_DATA_DIR}")
            else:
                logger.info(f"{destination} already exists; skipping move from extracted directory to {RAW_DATA_DIR}.")

        # Remove the now-empty subfolder.
        shutil.rmtree(subfolder)
        logger.info("Extracted directory cleaned-up (deleted).")
        logger.success(f"Dataset located in {RAW_DATA_DIR}, see {RAW_DATA_DIR}/readme.txt for some dataset info.")
        return
    elif not (RAW_DATA_DIR / "mitbih_data.zip").exists():
        logger.info("No dataset .zip file found, downloading from Kaggle with Kagglehub...")
        dataset_path = kagglehub.dataset_download('shayanfazeli/heartbeat')
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        for filename in os.listdir(dataset_path):
            shutil.copy(os.path.join(dataset_path, filename), RAW_DATA_DIR)
        logger.success(f"Dataset copied from Kaggle to {RAW_DATA_DIR}")

@app.command()
def main(
    train_input_path: Path = RAW_DATA_DIR / "mitbih_train.csv",
    train_output_path: Path = PROCESSED_DATA_DIR / "mitbih_train_cropped.csv",
    test_input_path: Path = RAW_DATA_DIR / "mitbih_test.csv",
    test_output_path: Path = PROCESSED_DATA_DIR / "mitbih_test_cropped.csv",
):
    """Main function to download ECG datasets."""
    setup_raw_ecg_dataset()

if __name__ == "__main__":
    app()
