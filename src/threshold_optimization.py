import numpy as np
import joblib
from data_loader import load_and_split_data
from business_metrics import business_cost

model = joblib.load("../models/fraud_model.pkl")

X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(
    "../data/creditcard.csv"
)

val_probs = model.predict_proba(X_val)[:, 1]

thresholds = np.linspace(0.01, 0.99, 100)
costs = []

for t in thresholds:
    preds = (val_probs > t).astype(int)
    costs.append(business_cost(y_val, preds))

best_threshold = thresholds[np.argmin(costs)]
print("Optimal threshold:", best_threshold)
