# realtime_predict.py
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

MODEL_PATH = MODELS_DIR / "best_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

# SAME ORDER AS TRAINING
ALL_FEATURES = [
    'V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14',
    'V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28',
    'Amount'
]

def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

def predict_single(input_dict):
    model, scaler = load_artifacts()

    # fill missing features with 0
    for feat in ALL_FEATURES:
        if feat not in input_dict:
            input_dict[feat] = 0.0

    df = pd.DataFrame([input_dict])

    # FORCE EXACT SAME ORDER AS TRAINING
    df = df[ALL_FEATURES]

    # scale Amount
    df['Amount'] = scaler.transform(df[['Amount']])

    prob = model.predict_proba(df)[:, 1][0]
    pred = int(model.predict(df)[0])

    return pred, prob

def main():
    tx = {
        "V1": -1.2,
        "V2": 0.5,
        "V3": -0.7,
        "Amount": 200.0
    }

    pred, prob = predict_single(tx)

    print("Prediction:", pred)
    print("Fraud probability:", prob)

if __name__ == "__main__":
    main()

