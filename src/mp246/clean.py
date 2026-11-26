
"""Data cleaning utilities for the MP246 warehouse prediction project."""

from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


def remove_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows and reset the index."""
    return df.drop_duplicates().reset_index(drop=True).copy()


def replace_sentinel_values(
    df: pd.DataFrame,
    columns: Sequence[str] = ("shift_hour",),
    sentinels: Iterable[int | float] = (-999,),
) -> pd.DataFrame:
    """Replace sentinel values with NumPy NaN."""
    cleaned = df.copy()

    for column in columns:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(
                cleaned[column], errors="coerce"
            ).replace(list(sentinels), np.nan)

    return cleaned


def fix_unit_mixups(
    df: pd.DataFrame,
    shift_col: str = "shift",
    morning_group_value: str = "morning",
    target_col: str = "picker_exp_months",
    factor: float = 12.0,
) -> pd.DataFrame:
    """Correct Morning picker experience values."""
    cleaned = df.copy()

    if shift_col not in cleaned.columns or target_col not in cleaned.columns:
        return cleaned

    cleaned[target_col] = pd.to_numeric(
        cleaned[target_col], errors="coerce"
    ).astype("float64")

    mask = (
        cleaned[shift_col]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq(morning_group_value.lower())
    )

    cleaned.loc[mask, target_col] = (
        cleaned.loc[mask, target_col] / factor
    )

    return cleaned


def mask_stuck_sensor(
    df: pd.DataFrame,
    column: str = "order_weight_kg",
    stuck_value: float = 20.83,
    date_col: str | None = None,
    start_date: str | None = "2026-04-28",
    end_date: str | None = "2026-05-15",
) -> pd.DataFrame:
    """Mask the known stuck sensor readings."""
    cleaned = df.copy()

    if column not in cleaned.columns:
        return cleaned

    if date_col and date_col in cleaned.columns:
        cleaned[column] = pd.to_numeric(
            cleaned[column], errors="coerce"
        ).astype("float64")

        dates = pd.to_datetime(
            cleaned[date_col], errors="coerce"
        )

        if start_date and end_date:
            mask = (
                dates.between(
                    pd.Timestamp(start_date),
                    pd.Timestamp(end_date),
                    inclusive="both",
                )
                & cleaned[column].eq(stuck_value)
            )

            cleaned.loc[mask, column] = np.nan

    return cleaned


def normalize_text_categories(
    df: pd.DataFrame,
    category_map: Mapping[str, Mapping[str, str]] | None = None,
) -> pd.DataFrame:
    """Strip whitespace and optionally normalize category names."""
    cleaned = df.copy()

    for column in cleaned.select_dtypes(
        include=["object", "string"]
    ).columns:

        cleaned[column] = cleaned[column].map(
            lambda value: value.strip()
            if isinstance(value, str)
            else value
        )

        if category_map and column in category_map:
            mapping = {
                str(key).strip().lower(): value
                for key, value in category_map[column].items()
            }

            cleaned[column] = cleaned[column].map(
                lambda value: mapping.get(
                    value.lower(), value
                )
                if isinstance(value, str)
                else value
            )

    return cleaned


def drop_leaking_columns(
    df: pd.DataFrame,
    columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Drop explicitly specified leaking columns."""
    existing = [
        column for column in columns
        if column in df.columns
    ]

    return df.drop(columns=existing).copy()


def clean_warehouse_data(
    df: pd.DataFrame,
    drop_duplicates: bool = True,
    sentinel_columns: Sequence[str] = ("shift_hour",),
    sentinels: Iterable[int | float] = (-999,),
    shift_col: str = "shift",
    morning_group_value: str = "morning",
    picker_exp_col: str = "picker_exp_months",
    morning_exp_factor: float = 12.0,
    stuck_col: str = "order_weight_kg",
    stuck_value: float = 20.83,
    date_col: str | None = None,
    stuck_start_date: str | None = "2026-04-28",
    stuck_end_date: str | None = "2026-05-15",
    category_map: Mapping[str, Mapping[str, str]] | None = None,
    leak_cols: Sequence[str] = (),
) -> pd.DataFrame:
    """Apply all warehouse data-cleaning operations."""
    cleaned = df.copy()

    if drop_duplicates:
        cleaned = remove_duplicate_rows(cleaned)

    cleaned = replace_sentinel_values(
        cleaned,
        columns=sentinel_columns,
        sentinels=sentinels,
    )

    cleaned = fix_unit_mixups(
        cleaned,
        shift_col=shift_col,
        morning_group_value=morning_group_value,
        target_col=picker_exp_col,
        factor=morning_exp_factor,
    )

    cleaned = mask_stuck_sensor(
        cleaned,
        column=stuck_col,
        stuck_value=stuck_value,
        date_col=date_col,
        start_date=stuck_start_date,
        end_date=stuck_end_date,
    )

    cleaned = normalize_text_categories(
        cleaned,
        category_map=category_map,
    )

    cleaned = drop_leaking_columns(
        cleaned,
        columns=leak_cols,
    )

    return cleaned
