# step5_train_models.py
from pathlib import Path
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

def train_models():
    # Load split data
    X_train, X_test, y_train, y_test = joblib.load(MODELS_DIR / "data_splits.pkl")

    models = {}

    # Logistic Regression
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(X_train, y_train)
    models["logistic"] = lr

    # Random Forest
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    rf.fit(X_train, y_train)
    models["random_forest"] = rf

    # XGBoost
    print("\nTraining XGBoost...")
    xgb = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    models["xgboost"] = xgb

    # Evaluate each model
    results = {}
    for name, model in models.items():
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:,1]

        auc = roc_auc_score(y_test, probs)
        print(f"\n{name} ROC-AUC: {auc:.6f}")

        report = classification_report(y_test, preds)
        print(report)

        results[name] = {"auc": auc, "report": report}

    # Save the models
    joblib.dump(models, MODELS_DIR / "trained_models.pkl")
    print("\n✔️ All models saved to:", MODELS_DIR / "trained_models.pkl")

    return results

if __name__ == "__main__":
    train_models()
