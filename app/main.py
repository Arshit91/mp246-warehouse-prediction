from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="MP246 Warehouse Picking Time API")

MODEL_PATH = Path("models/model.joblib")
_cached_model: Any = None


class PredictRequest(BaseModel):
    payload: dict[str, Any]


def _load_model() -> Any:
    global _cached_model
    if _cached_model is None and MODEL_PATH.exists():
        _cached_model = joblib.load(MODEL_PATH)
    return _cached_model


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest) -> dict[str, float]:
    model = _load_model()
    if model is None:
        return {"predicted_pick_seconds": 45.0}
    df = pd.DataFrame([request.payload])
    prediction = float(model.predict(df)[0])
    return {"predicted_pick_seconds": prediction}