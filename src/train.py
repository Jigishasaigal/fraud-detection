import os
import joblib
from lightgbm import LGBMClassifier
from data_loader import load_and_split_data
from config import MODEL_PARAMS

MODEL_DIR = "../models"
MODEL_PATH = os.path.join(MODEL_DIR, "fraud_model.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(
    "../data/creditcard.csv"
)

model = LGBMClassifier(**MODEL_PARAMS)
model.fit(X_train, y_train)

joblib.dump(model, MODEL_PATH)
print(f"Model saved at {MODEL_PATH}")
