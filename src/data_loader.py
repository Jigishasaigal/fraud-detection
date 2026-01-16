import pandas as pd
from sklearn.model_selection import train_test_split
from config import RANDOM_STATE, TEST_SIZE, VAL_SIZE

def load_and_split_data(path):
    df = pd.read_csv(path)
    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=TEST_SIZE + VAL_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=RANDOM_STATE
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
