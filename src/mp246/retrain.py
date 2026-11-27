from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import mlflow
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.mp246.train import (
    build_candidates,
    build_preprocessor,
    call_cleaner,
    log_sklearn_model,
)

LATER_PATH = Path("data/raw/later.csv")
REPORT_PATH = Path("reports/retrain_report.json")


def compute_combined_sha256(train_path: Path, later_path: Path) -> str:
    h = hashlib.sha256()
    h.update(train_path.read_bytes())
    h.update(later_path.read_bytes())
    return h.hexdigest()


def run_retrain(params_path: str = "params.yaml") -> dict[str, Any]:
    with open(params_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    train_path = Path(params["data"]["train_path"])
    target_col = params["data"].get("target_col", "pick_seconds")
    test_size = params["data"].get("test_size", 0.2)
    random_state = params["data"].get("random_state", 42)
    cleaning_cfg = params.get("cleaning", {})
    model_out_path = Path(params["output"].get("model_path", "models/model.joblib"))

    tracking_uri = params["mlflow"].get("tracking_uri", "sqlite:///mlflow.db")
    experiment_name = params["mlflow"].get("experiment_name", "mp246_warehouse_picking")
    registered_name = params["mlflow"].get("registered_model_name", "mp246_model")
    champion_alias = params["mlflow"].get("champion_alias", "champion")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    raw_train = pd.read_csv(train_path)
    raw_later = pd.read_csv(LATER_PATH)

    clean_train = call_cleaner(raw_train, cleaning_cfg)
    clean_later = call_cleaner(raw_later, cleaning_cfg)

    clean_train[target_col] = pd.to_numeric(clean_train[target_col], errors="coerce")
    clean_later[target_col] = pd.to_numeric(clean_later[target_col], errors="coerce")
    clean_train = clean_train.dropna(subset=[target_col]).reset_index(drop=True)
    clean_later = clean_later.dropna(subset=[target_col]).reset_index(drop=True)

    drop_cols = [c for c in [target_col, "record_id", "date"] if c in clean_train.columns]
    X_later_all = clean_later.drop(columns=drop_cols)
    y_later_all = clean_later[target_col]

    previous_model = joblib.load(model_out_path)
    before_retrain_later_mae = float(
        mean_absolute_error(y_later_all, previous_model.predict(X_later_all))
    )

    combined_df = pd.concat([clean_train, clean_later], ignore_index=True)
    X_all = combined_df.drop(columns=drop_cols)
    y_all = combined_df[target_col]
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_all, y_all, test_size=test_size, random_state=random_state
    )

    data_sha256 = compute_combined_sha256(train_path, LATER_PATH)

    best_mae = float("inf")
    best_run_name: str | None = None
    best_run_id: str | None = None
    best_pipeline: Pipeline | None = None

    for run_name, estimator, extra_params in build_candidates(params):
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_tr)),
                ("regressor", estimator),
            ]
        )
        with mlflow.start_run(run_name=f"retrain_{run_name}") as run:
            mlflow.log_param("stage", "stage4_retrain")
            mlflow.log_param("data_sha256", data_sha256)
            mlflow.log_param("train_rows", len(clean_train))
            mlflow.log_param("later_rows", len(clean_later))
            for k, v in extra_params.items():
                mlflow.log_param(k, v)

            pipeline.fit(X_tr, y_tr)
            val_preds = pipeline.predict(X_val)
            mae = float(mean_absolute_error(y_val, val_preds))
            after_later_mae = float(
                mean_absolute_error(y_later_all, pipeline.predict(X_later_all))
            )
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("later_mae", after_later_mae)
            log_sklearn_model(pipeline, artifact_path="model")

            if mae < best_mae:
                best_mae = mae
                best_run_name = f"retrain_{run_name}"
                best_run_id = run.info.run_id
                best_pipeline = pipeline

    if best_run_id is None or best_pipeline is None:
        raise RuntimeError("No retraining runs completed.")

    mv = mlflow.register_model(model_uri=f"runs:/{best_run_id}/model", name=registered_name)
    client = MlflowClient(tracking_uri=tracking_uri)
    client.set_registered_model_alias(registered_name, champion_alias, mv.version)

    model_out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, model_out_path)

    final_later_mae = float(
        mean_absolute_error(y_later_all, best_pipeline.predict(X_later_all))
    )
    report: dict[str, Any] = {
        "registered_model": registered_name,
        "champion_alias": champion_alias,
        "new_model_version": str(mv.version),
        "best_run_name": best_run_name,
        "best_run_id": best_run_id,
        "data_sha256": data_sha256,
        "clean_train_rows": int(len(clean_train)),
        "clean_later_rows": int(len(clean_later)),
        "combined_rows": int(len(combined_df)),
        "before_retrain_later_mae": round(before_retrain_later_mae, 4),
        "after_retrain_later_mae": round(final_later_mae, 4),
        "best_val_mae": round(best_mae, 4),
        "model_path": str(model_out_path),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run_retrain(), indent=2))