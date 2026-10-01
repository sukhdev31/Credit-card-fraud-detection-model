# step1_load_data.py
from pathlib import Path
import pandas as pd

# project root = parent of 'src'
ROOT = Path(__file__).resolve().parents[1]

# path to dataset
DATA_PATH = ROOT / "data" / "creditcard.csv"

def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Put creditcard.csv into /data folder.")
    df = pd.read_csv(DATA_PATH)
    return df

def main():
    df = load_data()
    print("Dataset loaded successfully!")
    print("Shape:", df.shape)
    print("First 5 rows:")
    print(df.head().to_string(index=False))

if __name__ == "__main__":
    main()

