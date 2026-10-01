# step4_split_data.py
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from step3_preprocess import preprocess

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

def split_data(test_size=0.2, random_state=42):
    X, y = preprocess()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    joblib.dump((X_train, X_test, y_train, y_test), MODELS_DIR / "data_splits.pkl")

    print("Train/Test split complete!")
    print("Shapes:")
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)
    print("Saved to:", MODELS_DIR / "data_splits.pkl")

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    split_data()
