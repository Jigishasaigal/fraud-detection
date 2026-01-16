RANDOM_STATE = 42

TEST_SIZE = 0.2
VAL_SIZE = 0.2

FRAUD_LOSS = 5000
FALSE_POSITIVE_COST = 200

MODEL_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "class_weight": {0: 1, 1: 25},
    "random_state": RANDOM_STATE
}
