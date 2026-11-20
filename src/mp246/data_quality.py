
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "raw"

train = pd.read_csv(DATA / "train.csv")
later = pd.read_csv(DATA / "later.csv")
holdout = pd.read_csv(DATA / "holdout_features.csv")

print("\n========== DATASET SHAPES ==========")
print("Train:", train.shape)
print("Later:", later.shape)
print("Holdout:", holdout.shape)

print("\n========== MISSING VALUES ==========")
print(train.isnull().sum())

print("\n========== DUPLICATE ROWS ==========")
print("Exact duplicate rows:", train.duplicated().sum())

print("\n========== SENTINEL VALUES ==========")
print("shift_hour == -999:", (train["shift_hour"] == -999).sum())

print("\n========== UNIT MIXUP INVESTIGATION ==========")
morning = train.loc[train["shift"] == "Morning", "picker_exp_months"]
print("Morning statistics:")
print(morning.describe())
print("\nMorning values above 60:", (morning > 60).sum())
print("Morning values divisible by 30:", (morning % 30 == 0).sum())

print("\nLater picker experience statistics:")
print(later["picker_exp_months"].describe())

print("\n========== STUCK SENSOR INVESTIGATION ==========")
train["date"] = pd.to_datetime(train["date"])
daily_weights = train.groupby("date")["order_weight_kg"].nunique()
stuck_dates = daily_weights[daily_weights == 1]
print("Dates with only one unique daily weight:")
print(stuck_dates)

print("\n========== TARGET INVESTIGATION ==========")
print(train["pick_seconds"].describe())
print("Later target statistics:")
print(later["pick_seconds"].describe())

print("\n========== CATEGORY INVESTIGATION ==========")
for column in ["zone_type", "shift"]:
    print(f"\n{column}:")
    print(train[column].value_counts(dropna=False))

print("\n========== COLUMN CHECK ==========")
print("Train columns:", train.columns.tolist())
print("Later columns:", later.columns.tolist())
print("Holdout columns:", holdout.columns.tolist())

print("\n========== AUDIT COMPLETE ==========")
