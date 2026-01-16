import joblib
import numpy as np
from data_loader import load_and_split_data
from business_metrics import business_cost

model = joblib.load("../models/fraud_model.pkl")

X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(
    "../data/creditcard.csv"
)

OPTIMAL_THRESHOLD = 0.01

test_probs = model.predict_proba(X_test)[:, 1]
test_preds = (test_probs > OPTIMAL_THRESHOLD).astype(int)


model_cost = business_cost(y_test, test_preds)

baseline_preds = np.zeros_like(y_test)
baseline_cost = business_cost(y_test, baseline_preds)

improvement_pct = (baseline_cost - model_cost) / baseline_cost * 100

print("Baseline Cost:", baseline_cost)
print("Model Cost:", model_cost)
print(f"Loss Reduction (%): {improvement_pct:.2f}")
