import pandas as pd
import numpy as np

def analyze_data():
    df = pd.read_csv('data/raw/train.csv')
    
    print("--- AUTOMATED DATA INSPECTION FOR FINDINGS.YAML ---")
    
    # 1. Duplicate rows
    dups = df.duplicated().sum()
    print(f"Duplicate rows extra copies: {dups}")
    
    # 2. Sentinel value in shift_hour
    sentinels = (df['shift_hour'] == -999).sum()
    print(f"Shift hour -999 sentinel count: {sentinels}")
    
    # 3. Unit mixup in picker_exp_months
    morning_max = df[df['shift'] == 'Morning']['picker_exp_months'].max()
    print(f"Morning shift max picker experience: {morning_max} (indicates factor 12)")
    
    # 4. Stuck sensor in order_weight_kg
    df_sorted = df.sort_values('date')
    stuck_rows = df_sorted[df_sorted['order_weight_kg'] == 20.83]
    print(f"Stuck sensor weight 20.83 from {stuck_rows['date'].min()} to {stuck_rows['date'].max()}")

if __name__ == "__main__":
    analyze_data()