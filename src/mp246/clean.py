from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


def remove_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows while preserving original row order."""
    return df.drop_duplicates().reset_index(drop=True)


def replace_sentinel_values(
    df: pd.DataFrame,
    columns: Sequence[str] = ("shift_hour",),
    sentinels: Iterable[int | float] = (-999, -1, 999, 9999),
) -> pd.DataFrame:
    """Replace sentinel placeholder values in specified numeric columns with np.nan."""
    cleaned = df.copy()
    sentinel_set = set(sentinels)
    for col in columns:
        if col in cleaned.columns:
            numeric_col = pd.to_numeric(cleaned[col], errors="coerce")
            cleaned[col] = numeric_col.where(~numeric_col.isin(sentinel_set), np.nan)
    return cleaned


def fix_unit_mixups(
    df: pd.DataFrame,
    group_col: str,
    group_value: str,
    target_col: str,
    factor: float,
    divide: bool = True,
) -> pd.DataFrame:
    """Adjust a column for a specific group by an exact conversion factor."""
    cleaned = df.copy()
    if group_col in cleaned.columns and target_col in cleaned.columns:
        mask = cleaned[group_col].astype("string").str.strip().str.lower() == group_value.strip().lower()
        numeric_vals = pd.to_numeric(cleaned.loc[mask, target_col], errors="coerce")
        if divide:
            cleaned.loc[mask, target_col] = numeric_vals / factor
        else:
            cleaned.loc[mask, target_col] = numeric_vals * factor
    return cleaned


def mask_stuck_sensor(
    df: pd.DataFrame,
    col: str = "order_weight_kg",
    stuck_value: float = 20.83,
    date_col: str | None = None,
    start_date: str | None = "2026-04-28",
    end_date: str | None = "2026-05-15",
    min_run_length: int = 2,
) -> pd.DataFrame:
    """Replace stuck sensor readings over a date window or repeated run with np.nan."""
    cleaned = df.copy()
    if col not in cleaned.columns:
        return cleaned

    numeric_vals = pd.to_numeric(cleaned[col], errors="coerce")
    is_stuck_val = np.isclose(numeric_vals, stuck_value, atol=1e-6, equal_nan=False)

    if date_col and date_col in cleaned.columns and start_date and end_date:
        dates = pd.to_datetime(cleaned[date_col], errors="coerce")
        in_window = (dates >= pd.to_datetime(start_date)) & (dates <= pd.to_datetime(end_date))
        cleaned.loc[is_stuck_val & in_window, col] = np.nan
        return cleaned

    # Fallback if no date column is passed: mask consecutive runs of the stuck value
    run_id = (numeric_vals != numeric_vals.shift()).cumsum()
    run_counts = numeric_vals.groupby(run_id).transform("count")
    cleaned.loc[is_stuck_val & (run_counts >= min_run_length), col] = np.nan
    return cleaned


def normalize_text_categories(
    df: pd.DataFrame,
    category_map: Mapping[str, Mapping[str, str]] | None = None,
) -> pd.DataFrame:
    """Normalize whitespace and explicit category spelling variants."""
    cleaned = df.copy()
    if not category_map:
        return cleaned
    for col, mapping in category_map.items():
        if col in cleaned.columns:
            series = cleaned[col].astype("string").str.strip()
            cleaned[col] = series.replace(mapping)
    return cleaned


def drop_leaking_columns(
    df: pd.DataFrame,
    leak_cols: Sequence[str] = (),
) -> pd.DataFrame:
    """Drop post-event or target-leaking columns not available at inference."""
    cols_to_drop = [c for c in leak_cols if c in df.columns]
    return df.drop(columns=cols_to_drop)


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
    """Run warehouse data cleaning steps in a deterministic order."""
    out = df.copy()

    if leak_cols:
        out = drop_leaking_columns(out, leak_cols=leak_cols)

    if drop_duplicates:
        out = remove_duplicate_rows(out)

    if sentinel_columns:
        out = replace_sentinel_values(out, columns=sentinel_columns, sentinels=sentinels)

    if shift_col in out.columns and picker_exp_col in out.columns:
        out = fix_unit_mixups(
            out,
            group_col=shift_col,
            group_value=morning_group_value,
            target_col=picker_exp_col,
            factor=morning_exp_factor,
            divide=True,
        )

    out = mask_stuck_sensor(
        out,
        col=stuck_col,
        stuck_value=stuck_value,
        date_col=date_col,
        start_date=stuck_start_date,
        end_date=stuck_end_date,
    )

    if category_map:
        out = normalize_text_categories(out, category_map=category_map)

    return out