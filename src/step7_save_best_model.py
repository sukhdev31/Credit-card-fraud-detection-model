# step7_save_best_model.py
from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

def save_best():
    src = MODELS_DIR / "rf_smote.pkl"
    dst = MODELS_DIR / "best_model.pkl"
    model = joblib.load(src)
    joblib.dump(model, dst)
    print(f"✔️ Best model saved to: {dst}")

if __name__ == "__main__":
    save_best()
