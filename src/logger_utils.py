# logger_utils.py
from pathlib import Path
import pandas as pd
import datetime

ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

LOG_PATH = LOGS_DIR / "suspicious_transactions.csv"

def log_suspicious(tx_dict):
    """
    tx_dict must contain: Amount, prediction, probability (and any other info)
    """

    df = pd.DataFrame([tx_dict])
    df["logged_at"] = datetime.datetime.now().isoformat()

    if LOG_PATH.exists():
        df.to_csv(LOG_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(LOG_PATH, index=False)

    return LOG_PATH
