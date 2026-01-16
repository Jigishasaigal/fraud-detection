import shap
import joblib
from data_loader import load_and_split_data

model = joblib.load("../models/fraud_model.pkl")

X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(
    "../data/creditcard.csv"
)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test[:1000])

shap.summary_plot(shap_values, X_test[:1000])
