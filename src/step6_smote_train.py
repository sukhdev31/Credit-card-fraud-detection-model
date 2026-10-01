# step6_smote_train.py
from pathlib import Path
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

def smote_and_train():
    print("Loading data...")
    X_train, X_test, y_train, y_test = joblib.load(MODELS_DIR / "data_splits.pkl")

    print("Applying SMOTE...")
    sm = SMOTE(random_state=42)
    X_resampled, y_resampled = sm.fit_resample(X_train, y_train)

    print("\nClass distribution after SMOTE:")
    print(y_resampled.value_counts())

    print("\nTraining Random Forest after SMOTE...")
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_resampled, y_resampled)

    # Evaluate
    preds = rf.predict(X_test)
    probs = rf.predict_proba(X_test)[:,1]

    auc = roc_auc_score(y_test, probs)
    print(f"\nROC-AUC after SMOTE: {auc:.6f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    # Save the new model
    joblib.dump(rf, MODELS_DIR / "rf_smote.pkl")
    print("\n✔️ Model saved to:", MODELS_DIR / "rf_smote.pkl")

if __name__ == "__main__":
    smote_and_train()
