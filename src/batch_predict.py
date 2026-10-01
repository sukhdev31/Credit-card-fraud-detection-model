# batch_predict.py
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

MODEL_PATH = MODELS_DIR / "best_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

# features
ALL_FEATURES = [
    'V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14',
    'V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28',
    'Amount'
]

def batch_predict(csv_path):
    # load model + scaler
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # read batch data
    df = pd.read_csv(csv_path)

    # keep only relevant columns
    for col in ALL_FEATURES:
        if col not in df.columns:
            df[col] = 0.0  # fill missing features

    df = df[ALL_FEATURES]

    # scale Amount
    df['Amount'] = scaler.transform(df[['Amount']])

    # predictions
    probs = model.predict_proba(df)[:, 1]
    preds = model.predict(df)

    # add results to original dataframe
    df_out = pd.read_csv(csv_path).copy()
    df_out['fraud_pred'] = preds
    df_out['fraud_prob'] = probs

    # save to output CSV
    out_path = Path(csv_path).with_name(Path(csv_path).stem + "_predicted.csv")
    df_out.to_csv(out_path, index=False)

    print("✔️ Batch predictions saved to:", out_path)

if __name__ == "__main__":
    # example usage
    batch_predict("data/sample_transactions.csv")

