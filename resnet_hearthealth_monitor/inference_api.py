from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np

from resnet_hearthealth_monitor.modeling.inference import predict_segment
from resnet_hearthealth_monitor.config import DEVICE

app = FastAPI(title="Mitbih ECG Inference API")

# Define the request model (assuming your input is a list of 125 features)
class InferenceRequest(BaseModel):
    segment: list  # Should be a list of 125 numbers

@app.get("/")
def read_root():
    return {"message": "Welcome to the Inference API. Use the /predict endpoint for predictions."}

@app.post("/predict")
async def predict(request: InferenceRequest):
    try:
        # Convert the input segment to a NumPy array
        segment = np.array(request.segment, dtype=np.float32)
        if segment.shape[0] != 125:
            raise HTTPException(status_code=400, detail="Input segment must have 125 features.")
        
        # Here, you'll need to load your ensemble models or use your pre-loaded models.
        predicted_class = predict_segment(segment, DEVICE)
        return {"predicted_class": predicted_class}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("inference_api:app", host="0.0.0.0", port=8000, reload=True)