import ast
import typer
from typing import Optional
from loguru import logger
import torch
from statistics import mode
import random
from pathlib import Path
from resnet_hearthealth_monitor.config import PROJ_ROOT

from resnet_hearthealth_monitor.modeling.resnet import ResNet125
from resnet_hearthealth_monitor.modeling.train import load_best_model

app = typer.Typer()

def predict_segment(segment, device):
    """
    Given a single data segment (1D array-like of length 125), this function loads each fold's model
    from the best trial, performs inference, and returns the final predicted class via majority vote.
    
    Returns:
        final_prediction (int): The predicted class label.
    """

    # Convert the segment to a torch tensor and add a batch dimension.
    segment_tensor = torch.tensor(segment, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    if segment_tensor.shape != (1, 1, 125):
        raise ValueError("Input segment must be tensor of shape (1, 1, 125).")
    
    best_model = load_best_model()
    raw_paths = best_model.user_attrs["model_paths"] 
    if isinstance(raw_paths, dict):
        raw_paths = list(raw_paths.values())

    # Resolve any relative paths against BEST_MODEL_DIR
    model_paths = []
    for p_str in raw_paths:
        p = Path(p_str)
        if p.is_absolute():
            abs_p = p
        else:
            # drop all leading ".." components
            parts = list(p.parts)
            while parts and parts[0] == "..":
                parts.pop(0)
            # rebuild the path under PROJECT_ROOT
            rel_p = Path(*parts)
            abs_p = Path(PROJ_ROOT) / rel_p

        if not abs_p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {abs_p}")
        model_paths.append(abs_p)

    torch_model_class = ResNet125()

    fold_predictions = []
    logger.info("Performing inference with best model...")
    with torch.no_grad():
        for checkpoint_path in model_paths:
            # Initialize a new model instance for each fold
            model = torch_model_class.to(device)
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            model.eval()
            
            # Generate prediction and extract the predicted class (as integer)
            y_pred = model(segment_tensor).argmax(dim=1)
            fold_predictions.append(y_pred.item())
    
    # Aggregate predictions across folds using majority vote (mode)
    final_prediction = mode(fold_predictions)
    return final_prediction

@app.callback(invoke_without_command=True)
def main(
    segment: Optional[str] = typer.Option(
        None,
        "--segment", "-s",
        help="Optional 125-length list of floats; if omitted, a random one is generated"
    ),
    device: str  = typer.Option("cpu", "--device",  "-d", help="‘cpu’ or ‘cuda’")
):
    """
     Parse the segment (or randomly generate one), run inference, and print the final majority-vote.
    """
    # parse & run inference exactly as you do today…
    # ── decide whether to parse or generate randomly
    if segment is None:
        segment_list = [random.random() for _ in range(125)]
        logger.info("No segment provided; generated a random 125‑length segment to use.")
    else:
        try:
            segment_list = ast.literal_eval(segment)
            assert isinstance(segment_list, list) and len(segment_list) == 125
        except Exception:
            raise typer.BadParameter("`segment` must be a 125‑length Python list of floats.")

    final_prediction = predict_segment(segment_list, device)
    logger.success(f"Done. Prediction: {final_prediction}")
    typer.echo(final_prediction)
    

if __name__ == "__main__":
    app()
