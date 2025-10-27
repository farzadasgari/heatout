import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import *
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict

def extract_date_from_filename(filename: str) -> pd.Timestamp | None:
    try:
        return pd.to_datetime(filename)
    except ValueError:
        return None

def compute_daily_means(file_path: Path) -> Dict[str, float] | None:
    date_str = file_path.stem
    date = extract_date_from_filename(date_str)
    if date is None:
        print(f"Invalid date format in filename: {file_path}")
        return None
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None
    
    cols_to_average = df.columns[2:]
    df[cols_to_average] = df[cols_to_average].replace([np.inf, -np.inf], np.nan)
    means = df[cols_to_average].mean(skipna=True)
    
    if means.isna().all():
        print(f"No valid data in {file_path} after filtering inf and nan.")
        return None
    
    row = {'Date': date}
    row.update(means.to_dict())
    
    return row

def merge_csv_data(root_dir: str = '../dataset/era5/csv', output_file: str = '../dataset/era.csv') -> None:
    script_dir = Path(__file__).parent
    root_path = (script_dir / root_dir).resolve()
    
    if not root_path.exists() or not root_path.is_dir():
        raise ValueError(f"Root directory '{root_path}' does not exist or is not a directory.")
    
    output_path = (script_dir / output_file).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    csv_files: List[Path] = sorted(root_path.rglob('*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in '{root_path}'.")
        return
    
    data: List[Dict[str, float]] = []
    for file in csv_files:
        row = compute_daily_means(file)
        if row:
            data.append(row)
    
    if not data:
        print("No valid data found after processing all files.")
        return
    
    final_df = pd.DataFrame(data)
    final_df = final_df.sort_values('Date').reset_index(drop=True)
    
    expected_columns = ["Date"] + bands
    final_df = final_df[expected_columns]
    
    final_df.to_csv(output_path, index=False)
    print(f"Merged data saved to '{output_path}' with {len(final_df)} rows.")

merge_csv_data()
