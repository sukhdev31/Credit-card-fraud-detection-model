from logger_utils import log_suspicious

# Example suspicious transaction
tx = {
    "TransactionID": 123,
    "Amount": 5000,
    "prediction": 1,
    "probability": 0.97
}

path = log_suspicious(tx)
print("Logged to:", path)

