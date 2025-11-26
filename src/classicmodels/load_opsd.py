"""
load_opsd.py
------------
Load tidy per-country CSVs (BE, DK, NL)
Each CSV contains: timestamp, load, wind (optional), solar (optional)
Uses the provided data loading logic from the reference repository
"""

import pandas as pd
import yaml
import os
import warnings

warnings.filterwarnings('ignore')


def load_tidy_country_csvs_from_config(config_path='src/config.yaml'):
    """
    Read 3 separate tidy CSVs per country as per config settings
    Each CSV contains timestamp, load, wind (optional), solar (optional)
    
    Uses the provided data loading logic:
    - Drop optional 'cet_cest_timestamp' column
    - Rename columns to standard names
    - Convert timestamp to datetime
    - Drop rows with missing load
    - Sort by timestamp
    
    Returns: {country_code: DataFrame}
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    countries = config['countries']
    input_folder = config['dataset']['input_folder']
    dfs = {}

    print("\n" + "="*80)
    print("LOADING COUNTRY DATA FROM CSV FILES")
    print("="*80)

    for cc in countries:
        file_path = os.path.join(input_folder, f"{cc}.csv")

        if not os.path.exists(file_path):
            print(f"\n❌ Warning: file for country '{cc}' not found at {file_path}, skipping")
            continue

        print(f"\n🌍 Loading {cc} from {file_path}...")
        
        df = pd.read_csv(file_path)
        
        print(f"   📊 Raw shape: {df.shape}")
        print(f"      Columns: {list(df.columns)[:10]}...")

        # Safely drop optional column
        df = df.drop(columns=['cet_cest_timestamp'], errors='ignore')

        # Rename known columns to standard names (missing keys are ignored)
        df.rename(columns={
            'utc_timestamp': 'timestamp',
            f'{cc}_load_actual_entsoe_transparency': 'load',
            f'{cc}_solar_generation_actual': 'solar',
            f'{cc}_wind_generation_actual': 'wind'
        }, inplace=True)

        # Ensure timestamp exists and convert to datetime
        if 'timestamp' not in df.columns:
            raise ValueError(
                f"Missing column 'timestamp' — file {file_path} columns: {list(df.columns)}"
            )

        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

        # Ensure load exists
        if 'load' not in df.columns:
            raise ValueError(
                f"No 'load' column found in {file_path}. Columns: {list(df.columns)}"
            )

        # Drop rows with missing load
        before_drop = len(df)
        df.dropna(subset=['load'], inplace=True)
        after_drop = len(df)
        
        if before_drop > after_drop:
            print(f"   🗑️  Dropped {before_drop - after_drop} rows with NaN in load")

        # Sort by timestamp
        df.sort_values('timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)

        dfs[cc] = df

        print(f"   ✅ Loaded {len(df)} rows")
        print(f"      Columns: {list(df.columns)}")
        print(f"      Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    print("\n" + "="*80)
    print(f"✅ Successfully loaded {len(dfs)} countries: {list(dfs.keys())}")
    print("="*80 + "\n")

    return dfs


if __name__ == '__main__':
    # Load data using config
    dfs = load_tidy_country_csvs_from_config()
    
    # Print sample data
    for cc, df in dfs.items():
        print(f"\n{cc} - First 5 rows:")
        print(df.head())
        print(f"\n{cc} - Statistics:")
        print(df.describe())
