💳 Credit Card Fraud Detection System

A machine learning system that detects fraudulent credit card transactions using multiple models, SMOTE balancing, and a Streamlit web app.

---

## 🚀 Features

- Model training (Logistic Regression, Random Forest, XGBoost)
- SMOTE oversampling to handle class imbalance
- Best model saved (`best_model.pkl`)
- Realtime prediction
- Batch CSV prediction
- Suspicious transaction logging
- Streamlit web dashboard

---

## 🧠 Machine Learning Pipeline

1. Load dataset
2. Exploratory Data Analysis (EDA)
3. Preprocessing & feature scaling
4. Train/Test split
5. Train models
6. Apply SMOTE & retrain
7. Select best model
8. Save best model to `models/best_model.pkl`

---

## 📁 Project Structure

credit_fraud_project/
├─ data/ # dataset and sample CSV
├─ logs/ # suspicious transaction logs
├─ models/ # saved models (scaler + best model)
├─ src/ # all Python code
│ ├─ step1_load_data.py
│ ├─ step2_eda.py
│ ├─ step3_preprocess.py
│ ├─ step4_split_data.py
│ ├─ step5_train_models.py
│ ├─ step6_smote_train.py
│ ├─ step7_save_best_model.py
│ ├─ realtime_predict.py
│ ├─ batch_predict.py
│ ├─ logger_utils.py
│ ├─ streamlit_app.py
├─ venv/
├─ requirements.txt
├─ README.md
└─ run_app.bat

yaml
Copy code

---

## ▶️ How to Run the App

1️⃣ Open terminal and activate virtual environment:

venv\Scripts\activate

yaml
Copy code

2️⃣ Run Streamlit:

streamlit run src\streamlit_app.py

yaml
Copy code

---

## ⚡ One-click Launcher

Double-click:

run_app.bat

yaml
Copy code

This will:

- activate environment
- start Streamlit app
- open in browser

---

## 📦 Batch Prediction

Place a CSV file inside `data/` (example):

data/sample_transactions.csv

makefile
Copy code

Run:

python src\batch_predict.py

csharp
Copy code

Results saved as:

sample_transactions_predicted.csv

yaml
Copy code

---

## 🧾 Logging Suspicious Transactions

Whenever a prediction is fraud (`pred = 1`), it is logged:

logs/suspicious_transactions.csv

yaml
Copy code

Includes:

- transaction data
- fraud probability
- timestamp

---

## 💻 Technologies Used

- Python
- Pandas
- Scikit-Learn
- XGBoost
- imbalanced-learn
- Streamlit
- Joblib

---

## 👨‍💻 Author

Sukhdev Chhabra


---

## 🏁 Status

✔️ Fully working  
✔️ Complete project  
✔️ Ready for demo or submission