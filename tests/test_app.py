from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from src.mp246.clean import clean_warehouse_data


def test_clean_warehouse_data_drops_duplicates_and_replaces_sentinels() -> None:
    df = pd.DataFrame(
        {
            "date": ["2026-04-15", "2026-04-15", "2026-04-16"],
            "shift": ["Evening", "Evening", "Night"],
            "picker_exp_months": [10.0, 10.0, 14.0],
            "shift_hour": [-999, -999, 21],
            "order_weight_kg": [6.5, 6.5, 9.0],
        }
    )

    cleaned = clean_warehouse_data(
        df,
        drop_duplicates=True,
        sentinel_columns=("shift_hour",),
        sentinels=(-999,),
        shift_col="shift",
        morning_group_value="Morning",
        picker_exp_col="picker_exp_months",
        morning_exp_factor=12.0,
        stuck_col="order_weight_kg",
        date_col="date",
        stuck_start_date="2026-04-28",
        stuck_end_date="2026-05-15",
    )

    assert len(cleaned) == 2
    assert pd.isna(cleaned.loc[0, "shift_hour"])
    assert cleaned.loc[1, "shift_hour"] == 21


def test_clean_warehouse_data_scales_morning_experience_by_12() -> None:
    df = pd.DataFrame(
        {
            "date": ["2026-04-10", "2026-04-10"],
            "shift": ["Morning", "Night"],
            "picker_exp_months": [3.0, 3.0],
            "shift_hour": [9, 23],
            "order_weight_kg": [4.5, 7.0],
        }
    )

    cleaned = clean_warehouse_data(
        df,
        drop_duplicates=False,
        sentinel_columns=(),
        sentinels=(),
        shift_col="shift",
        morning_group_value="Morning",
        picker_exp_col="picker_exp_months",
        morning_exp_factor=12.0,
        stuck_col="order_weight_kg",
        date_col="date",
        stuck_start_date="2026-04-28",
        stuck_end_date="2026-05-15",
    )

    assert cleaned.loc[0, "picker_exp_months"] == pytest.approx(0.25)
    assert cleaned.loc[1, "picker_exp_months"] == pytest.approx(3.0)


def test_fastapi_health_and_predict_endpoints() -> None:
    client = TestClient(app)

    health_res = client.get("/health")
    assert health_res.status_code == 200
    assert health_res.json().get("status") == "ok"

    payload = {
        "payload": {
            "items_in_order": 4,
            "aisle_distance_m": 50.0,
            "picker_exp_months": 12.0,
            "shelf_height_m": 1.8,
            "shift_hour": 11,
            "order_weight_kg": 5.5,
            "zone_type": "Ambient",
            "shift": "Morning",
        }
    }
    predict_res = client.post("/predict", json=payload)
    assert predict_res.status_code == 200
    assert "predicted_pick_seconds" in predict_res.json()