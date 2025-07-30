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

def generate_cropped_dataset(input_path: Path, output_path: Path, crop_position: int = 125) -> pd.DataFrame:
    """
    Reads a CSV file from input_path, crops it to retain only the first `crop_position` columns 
    and the final column (assumed to be labels), saves the cropped DataFrame to output_path, 
    and returns the cropped DataFrame.
    
    Parameters:
        input_path (Path): Path to the input CSV file.
        output_path (Path): Path where the cropped CSV file will be saved.
        crop_position (int): Number of feature columns to keep from the start of the CSV.
                              (Default is 125.)
                              
    Returns:
        pd.DataFrame: The cropped DataFrame.
    
    Raises:
        FileNotFoundError: If the input CSV file does not exist.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file {input_path} does not exist.")
    
    # Load the CSV file. Assuming the file has no header.
    df = pd.read_csv(input_path, header=None)
        
    # Crop the DataFrame: keep columns 0 to crop_position-1 and the final column.
    cropped = pd.concat([df.iloc[:, :crop_position], df.iloc[:, -1]], axis=1)
    
    if output_path.exists():
        logger.info(f"{output_path} already exists, not overwriting.")
    else:
        # Save the cropped DataFrame to the output path without index or header.
        cropped.to_csv(output_path, index=False, header=False)
        logger.success(f"Processed {input_path.name} saved to {output_path}")
    return cropped


@app.command()
def main(
    train_input_path: Path = RAW_DATA_DIR / "mitbih_train.csv",
    train_output_path: Path = PROCESSED_DATA_DIR / "mitbih_train_cropped.csv",
    test_input_path: Path = RAW_DATA_DIR / "mitbih_test.csv",
    test_output_path: Path = PROCESSED_DATA_DIR / "mitbih_test_cropped.csv",
):
    """Main function to preprocess ECG datasets."""
    generate_cropped_dataset(train_input_path, train_output_path, crop_position=125)
    generate_cropped_dataset(test_input_path, test_output_path, crop_position=125)
    logger.success("All datasets processed successfully.")


if __name__ == "__main__":
    app()