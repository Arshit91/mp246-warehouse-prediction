from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.mp246.clean import clean_warehouse_data


def compute_sha256(file_path: str | Path) -> str:
    """Compute SHA-256 hex digest of the original file."""
    return hashlib.sha256(Path(file_path).read_bytes()).hexdigest()


def call_cleaner(df: pd.DataFrame, cleaning_cfg: dict[str, Any]) -> pd.DataFrame:
    """Call Paridhi's clean_warehouse_data with explicit MP246 settings."""
    return clean_warehouse_data(
        df,
        drop_duplicates=cleaning_cfg.get("drop_duplicates", True),
        sentinel_columns=tuple(cleaning_cfg.get("sentinel_columns", ("shift_hour",))),
        sentinels=tuple(cleaning_cfg.get("sentinels", (-999,))),
        shift_col=cleaning_cfg.get("shift_col", "shift"),
        morning_group_value=cleaning_cfg.get("morning_group_value", "Morning"),
        picker_exp_col=cleaning_cfg.get("picker_exp_col", "picker_exp_months"),
        morning_exp_factor=float(cleaning_cfg.get("morning_exp_factor", 12.0)),
        stuck_col=cleaning_cfg.get("stuck_col", "order_weight_kg"),
        date_col=cleaning_cfg.get("date_col", "date"),
        stuck_start_date=cleaning_cfg.get("stuck_start_date", "2026-04-28"),
        stuck_end_date=cleaning_cfg.get("stuck_end_date", "2026-05-15"),
    )


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build numeric and categorical preprocessing pipeline."""
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )


def build_candidates(params: dict[str, Any]) -> list[tuple[str, Any, dict[str, Any]]]:
    """Build 3+ candidate models/hyperparameter settings from params.yaml."""
    models_cfg = params.get("models", {})
    rs = params["data"].get("random_state", 42)

    ridge_cfg = models_cfg.get("ridge", {"alpha": 1.0})
    rf_cfg = models_cfg.get("random_forest", {"n_estimators": 100, "max_depth": 10})
    gbr_cfg = models_cfg.get(
        "gradient_boosting", {"n_estimators": 150, "learning_rate": 0.05, "max_depth": 4}
    )

    return [
        (
            "ridge_v1",
            Ridge(alpha=ridge_cfg.get("alpha", 1.0), random_state=rs),
            {"model_type": "Ridge", "alpha": ridge_cfg.get("alpha", 1.0)},
        ),
        (
            "random_forest_v1",
            RandomForestRegressor(
                n_estimators=rf_cfg.get("n_estimators", 100),
                max_depth=rf_cfg.get("max_depth", 10),
                random_state=rs,
                n_jobs=-1,
            ),
            {
                "model_type": "RandomForestRegressor",
                "n_estimators": rf_cfg.get("n_estimators", 100),
                "max_depth": rf_cfg.get("max_depth", 10),
            },
        ),
        (
            "gradient_boosting_v1",
            GradientBoostingRegressor(
                n_estimators=gbr_cfg.get("n_estimators", 150),
                learning_rate=gbr_cfg.get("learning_rate", 0.05),
                max_depth=gbr_cfg.get("max_depth", 4),
                random_state=rs,
            ),
            {
                "model_type": "GradientBoostingRegressor",
                "n_estimators": gbr_cfg.get("n_estimators", 150),
                "learning_rate": gbr_cfg.get("learning_rate", 0.05),
                "max_depth": gbr_cfg.get("max_depth", 4),
            },
        ),
    ]


def log_sklearn_model(model: Pipeline, artifact_path: str) -> None:
    """Log scikit-learn model, using cloudpickle fallback for MLflow 3.15 if needed."""
    try:
        mlflow.sklearn.log_model(model, artifact_path=artifact_path)
    except Exception as exc:
        if "untrusted" in str(exc).lower() or "pickle" in str(exc).lower():
            mlflow.sklearn.log_model(
                model,
                artifact_path=artifact_path,
                serialization_format="cloudpickle",
            )
        else:
            raise


def main(params_path: str = "params.yaml") -> None:
    with open(params_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    train_path = Path(params["data"]["train_path"])
    target_col = params["data"].get("target_col", "pick_seconds")
    test_size = params["data"].get("test_size", 0.2)
    random_state = params["data"].get("random_state", 42)
    cleaning_cfg = params.get("cleaning", {})

    data_sha256 = compute_sha256(train_path)

    tracking_uri = params["mlflow"].get("tracking_uri", "sqlite:///mlflow.db")
    experiment_name = params["mlflow"].get("experiment_name", "mp246_warehouse_picking")
    registered_name = params["mlflow"].get("registered_model_name", "mp246_model")
    champion_alias = params["mlflow"].get("champion_alias", "champion")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    raw_df = pd.read_csv(train_path)
    cleaned_df = call_cleaner(raw_df, cleaning_cfg)

    cleaned_df[target_col] = pd.to_numeric(cleaned_df[target_col], errors="coerce")
    cleaned_df = cleaned_df.dropna(subset=[target_col]).reset_index(drop=True)

    drop_cols = [c for c in [target_col, "record_id", "date"] if c in cleaned_df.columns]
    X = cleaned_df.drop(columns=drop_cols)
    y = cleaned_df[target_col]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    best_mae = float("inf")
    best_run_id: str | None = None
    best_pipeline: Pipeline | None = None

    for run_name, estimator, extra_params in build_candidates(params):
        preprocessor = build_preprocessor(X_train)
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", estimator),
            ]
        )

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_param("data_sha256", data_sha256)
            for k, v in extra_params.items():
                mlflow.log_param(k, v)

            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_val)
            mae = float(mean_absolute_error(y_val, preds))
            mlflow.log_metric("mae", mae)

            log_sklearn_model(pipeline, artifact_path="model")

            if mae < best_mae:
                best_mae = mae
                best_run_id = run.info.run_id
                best_pipeline = pipeline

    if best_run_id is None or best_pipeline is None:
        raise RuntimeError("No training runs completed.")

    model_uri = f"runs:/{best_run_id}/model"
    mv = mlflow.register_model(model_uri=model_uri, name=registered_name)
    client = MlflowClient()
    client.set_registered_model_alias(registered_name, champion_alias, mv.version)

    model_out_path = Path(params["output"].get("model_path", "models/model.joblib"))
    model_out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, model_out_path)

    print(f"Experiment: {experiment_name}")
    print(f"Best Run ID: {best_run_id} | Best Val MAE: {best_mae:.4f}")
    print(f"Registered Model: {registered_name} v{mv.version} -> @{champion_alias}")
    print(f"Saved model to: {model_out_path}")


if __name__ == "__main__":
    main()