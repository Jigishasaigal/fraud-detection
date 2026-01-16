from sklearn.metrics import confusion_matrix
from config import FRAUD_LOSS, FALSE_POSITIVE_COST

def business_cost(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fn * FRAUD_LOSS + fp * FALSE_POSITIVE_COST
