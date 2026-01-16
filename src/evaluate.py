import joblib
from sklearn.metrics import roc_auc_score, classification_report
from data_loader import load_and_split_data

model = joblib.load("../models/fraud_model.pkl")

X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(
    "../data/creditcard.csv"
)

test_probs = model.predict_proba(X_test)[:, 1]

print("ROC-AUC:", roc_auc_score(y_test, test_probs))


