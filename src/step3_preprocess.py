# step3_preprocess.py
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from step1_load_data import load_data

# project paths
ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

def preprocess():
    df = load_data().copy()

    # Drop Time column
    if "Time" in df.columns:
        df.drop(columns=["Time"], inplace=True)

    # Split features and target
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Scale the Amount column
    scaler = StandardScaler()
    X["Amount"] = scaler.fit_transform(X[["Amount"]])

    # Save the scaler for reuse
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

    return X, y

def main():
    X, y = preprocess()
    print("Preprocessing complete.")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Scaler saved to:", MODELS_DIR / "scaler.pkl")

if __name__ == "__main__":
    main()
