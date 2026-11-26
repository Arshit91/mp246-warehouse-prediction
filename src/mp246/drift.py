
"""Drift audit comparing cleaned training and later warehouse datasets."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.mp246.clean import clean_warehouse_data


ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = ROOT / "data" / "raw" / "train.csv"
LATER_PATH = ROOT / "data" / "raw" / "later.csv"
MODEL_PATH = ROOT / "models" / "model.joblib"
REPORT_PATH = ROOT / "reports" / "drift_report.json"

TARGET = "pick_seconds"

NUMERIC_FEATURES = [
    "items_in_order",
    "aisle_distance_m",
    "picker_exp_months",
    "shelf_height_m",
    "shift_hour",
    "order_weight_kg",
]

CATEGORICAL_FEATURES = [
    "zone_type",
    "shift",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

NUMERIC_THRESHOLD = 0.25
CATEGORICAL_THRESHOLD = 0.10


def load_and_clean(path: Path) -> pd.DataFrame:
    """Load a CSV file and apply the shared cleaning pipeline."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    return clean_warehouse_data(
        df,
        date_col="date",
    )


def prepare_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare exactly the eight features expected by the model."""
    missing = [
        column for column in MODEL_FEATURES
        if column not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing model features: {missing}")

    X = df[MODEL_FEATURES].copy()

    for column in NUMERIC_FEATURES:
        X[column] = pd.to_numeric(
            X[column], errors="coerce"
        ).astype("float64")

    for column in CATEGORICAL_FEATURES:
        X[column] = X[column].astype(object)
        X[column] = X[column].where(
            pd.notna(X[column]), np.nan
        )

    return X


def calculate_mae(model, df: pd.DataFrame) -> float:
    """Calculate mean absolute error."""
    X = prepare_model_features(df)

    y = pd.to_numeric(
        df[TARGET], errors="coerce"
    ).astype("float64")

    valid = y.notna()

    if not valid.any():
        raise ValueError(
            f"No valid target values found in {TARGET}"
        )

    predictions = model.predict(X.loc[valid])

    return float(
        np.mean(
            np.abs(
                y.loc[valid].to_numpy() - predictions
            )
        )
    )


def calculate_numeric_drift(
    train: pd.DataFrame,
    later: pd.DataFrame,
    columns: list[str],
) -> dict:
    """Compare numerical distributions using mean and standard deviation."""
    results = {}

    for column in columns:
        train_values = pd.to_numeric(
            train[column], errors="coerce"
        ).dropna()

        later_values = pd.to_numeric(
            later[column], errors="coerce"
        ).dropna()

        if train_values.empty or later_values.empty:
            results[column] = {
                "drift_detected": False,
                "reason": "Insufficient valid values",
            }
            continue

        train_mean = float(train_values.mean())
        later_mean = float(later_values.mean())

        train_std = float(train_values.std())
        later_std = float(later_values.std())

        mean_diff = abs(later_mean - train_mean)

        mean_shift_ratio = (
            mean_diff / max(abs(train_mean), 1e-9)
        )

        if train_std == 0:
            std_shift_ratio = (
                0.0 if later_std == 0 else float("inf")
            )
        else:
            std_shift_ratio = (
                abs(later_std - train_std) / train_std
            )

        drift_detected = (
            mean_shift_ratio > NUMERIC_THRESHOLD
            or std_shift_ratio > NUMERIC_THRESHOLD
        )

        results[column] = {
            "train_mean": train_mean,
            "later_mean": later_mean,
            "train_std": train_std,
            "later_std": later_std,
            "mean_diff": float(mean_diff),
            "mean_shift_ratio": float(mean_shift_ratio),
            "std_shift_ratio": float(std_shift_ratio),
            "threshold": NUMERIC_THRESHOLD,
            "drift_detected": bool(drift_detected),
        }

    return results


def calculate_categorical_drift(
    train: pd.DataFrame,
    later: pd.DataFrame,
    columns: list[str],
) -> dict:
    """Compare category proportions between datasets."""
    results = {}

    for column in columns:
        train_props = train[column].value_counts(
            normalize=True,
            dropna=False,
        )

        later_props = later[column].value_counts(
            normalize=True,
            dropna=False,
        )

        all_categories = train_props.index.union(
            later_props.index
        )

        differences = (
            later_props.reindex(
                all_categories, fill_value=0
            )
            - train_props.reindex(
                all_categories, fill_value=0
            )
        ).abs()

        max_diff = (
            float(differences.max())
            if len(differences)
            else 0.0
        )

        results[column] = {
            "train_proportions": {
                str(key): float(value)
                for key, value in train_props.items()
            },
            "later_proportions": {
                str(key): float(value)
                for key, value in later_props.items()
            },
            "max_proportion_difference": max_diff,
            "threshold": CATEGORICAL_THRESHOLD,
            "drift_detected": bool(
                max_diff > CATEGORICAL_THRESHOLD
            ),
        }

    return results


def get_drifted_features(results: dict) -> list[str]:
    """Return names of features that crossed the drift threshold."""
    return [
        name
        for name, result in results.items()
        if result.get("drift_detected", False)
    ]


def make_json_safe(value):
    """Convert NumPy values and non-finite numbers to JSON-safe values."""
    if isinstance(value, dict):
        return {
            key: make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            make_json_safe(item)
            for item in value
        ]

    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return None
        return float(value)

    if isinstance(value, np.integer):
        return int(value)

    return value


def main() -> None:
    """Run the drift audit and save the JSON report."""

    print("Loading and cleaning datasets...")

    train = load_and_clean(TRAIN_PATH)
    later = load_and_clean(LATER_PATH)

    print(f"Cleaned train rows: {len(train)}")
    print(f"Cleaned later rows: {len(later)}")

    if TARGET not in train.columns or TARGET not in later.columns:
        raise ValueError(
            f"Target column '{TARGET}' is missing."
        )

    print("Loading model...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    expected_features = list(
        getattr(model, "feature_names_in_", MODEL_FEATURES)
    )

    if expected_features != MODEL_FEATURES:
        raise ValueError(
            "Model feature order does not match configured features. "
            f"Model expects: {expected_features}"
        )

    print("Calculating MAE...")

    train_mae = calculate_mae(model, train)
    later_mae = calculate_mae(model, later)

    print("Calculating numerical feature drift...")

    numeric_drift = calculate_numeric_drift(
        train,
        later,
        NUMERIC_FEATURES,
    )

    print("Calculating target drift...")

    target_drift = calculate_numeric_drift(
        train,
        later,
        [TARGET],
    )

    print("Calculating categorical drift...")

    categorical_drift = calculate_categorical_drift(
        train,
        later,
        CATEGORICAL_FEATURES,
    )

    drifted_numeric = get_drifted_features(numeric_drift)
    drifted_categorical = get_drifted_features(categorical_drift)
    drifted_targets = get_drifted_features(target_drift)

    report = {
        "dataset_summary": {
            "train_rows_after_cleaning": len(train),
            "later_rows_after_cleaning": len(later),
            "train_columns": len(train.columns),
            "later_columns": len(later.columns),
        },
        "model_performance": {
            "train_mae": train_mae,
            "later_mae": later_mae,
            "mae_increase": later_mae - train_mae,
        },
        "numeric_drift": numeric_drift,
        "categorical_drift": categorical_drift,
        "target_drift": target_drift,
        "drifted_features": {
            "numeric": drifted_numeric,
            "categorical": drifted_categorical,
            "all": drifted_numeric + drifted_categorical,
        },
        "drifted_targets": drifted_targets,
    }

    report = make_json_safe(report)

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
            allow_nan=False,
        )

    print(f"\nDrift report saved to: {REPORT_PATH}")
    print(f"Train MAE: {train_mae:.2f}")
    print(f"Later MAE: {later_mae:.2f}")

    # Print calculated variables directly to avoid Pylance
    # errors from nested dictionary indexing.
    print(
        "Drifted input features:",
        drifted_numeric + drifted_categorical,
    )

    print(
        "Drifted targets:",
        drifted_targets,
    )


if __name__ == "__main__":
    main()
