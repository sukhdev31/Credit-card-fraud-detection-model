import streamlit as st
from pathlib import Path
import joblib
import pandas as pd
from logger_utils import log_suspicious

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

# Load model & scaler
MODEL = joblib.load(MODELS_DIR / "best_model.pkl")
SCALER = joblib.load(MODELS_DIR / "scaler.pkl")

ALL_FEATURES = [
    'V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14',
    'V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28',
    'Amount'
]

st.title("💳 Credit Card Fraud Detection App")

mode = st.radio("Select Mode", ["Realtime", "Batch CSV"])

# ========================
# REALTIME MODE
# ========================
if mode == "Realtime":
    st.subheader("Enter Transaction Features")

    vals = {}
    for f in ALL_FEATURES:
        vals[f] = st.number_input(f, value=0.0)

    if st.button("Predict"):
        df = pd.DataFrame([vals])
        df = df[ALL_FEATURES]
        df["Amount"] = SCALER.transform(df[["Amount"]])

        prob = MODEL.predict_proba(df)[:,1][0]
        pred = int(MODEL.predict(df)[0])

        st.write("Prediction:", pred)
        st.write("Fraud probability:", prob)

        if pred == 1 or prob >= 0.9:
            log_suspicious({**vals, "prediction": pred, "probability": prob})
            st.warning("⚠️ Suspicious transaction logged!")


# ========================
# BATCH CSV MODE
# ========================
else:
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded:
        df_in = pd.read_csv(uploaded)
        
        for col in ALL_FEATURES:
            if col not in df_in.columns:
                df_in[col] = 0.0

        df = df_in[ALL_FEATURES]
        df["Amount"] = SCALER.transform(df[["Amount"]])

        probs = MODEL.predict_proba(df)[:,1]
        preds = MODEL.predict(df)

        df_in["fraud_pred"] = preds
        df_in["fraud_prob"] = probs

        st.dataframe(df_in.head())

        suspicious = df_in[df_in["fraud_prob"] >= 0.9]
        if not suspicious.empty:
            for _, row in suspicious.iterrows():
                log_suspicious(row.to_dict())
            st.warning(f"⚠️ Logged {len(suspicious)} suspicious transactions!")

        st.success("Batch processing complete!")
