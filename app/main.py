
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.mp246.clean import clean_warehouse_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("mp246.api")

app = FastAPI(
    title="MP246 Warehouse Picking Time API",
    version="0.2.0",
)

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.joblib"
_cached_model: Any = None


class PredictRequest(BaseModel):
    payload: dict[str, Any] = Field(default_factory=dict)
    records: list[dict[str, Any]] | None = None


def _load_model() -> Any:
    global _cached_model

    if _cached_model is None:
        if not MODEL_PATH.exists():
            logger.error("Model missing: %s", MODEL_PATH)
            raise HTTPException(
                status_code=503,
                detail="Model artifact unavailable",
            )

        try:
            _cached_model = joblib.load(MODEL_PATH)
            logger.info("Model loaded from %s", MODEL_PATH)
        except Exception:
            logger.exception("Failed to load model")
            raise HTTPException(
                status_code=503,
                detail="Model could not be loaded",
            )

    return _cached_model


@app.get("/health")
def health() -> dict[str, str]:
    logger.info("Health check requested")
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest) -> dict[str, Any]:
    start = time.perf_counter()

    rows = request.records if request.records is not None else [request.payload]

    if not rows or any(not row for row in rows):
        raise HTTPException(
            status_code=422,
            detail="At least one non-empty feature record is required",
        )

    try:
        model = _load_model()

        raw_df = pd.DataFrame(rows)

        cleaned_df = clean_warehouse_data(
            raw_df,
            drop_duplicates=False,
            date_col="date",
        )

        features_df = cleaned_df.drop(
            columns=["pick_seconds", "record_id", "date"],
            errors="ignore",
        )

        predictions = [
            float(value)
            for value in model.predict(features_df)
        ]

    except HTTPException:
        raise
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=400,
            detail="Prediction failed; check feature values and model compatibility",
        )

    latency_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "Served %d predictions in %.2f ms",
        len(predictions),
        latency_ms,
    )

    return {
        "predicted_pick_seconds": predictions[0],
        "predictions": predictions,
    }
