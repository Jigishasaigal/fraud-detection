import pandas as pd
import numpy as np

from datetime import datetime, timedelta

np.random.seed(42)

# ---------------- CONFIG ----------------

n = 500

# ---------------- TIMESTAMPS ----------------

timestamps = [
    datetime.now() - timedelta(minutes=i * 10)
    for i in range(n)
]

timestamps.reverse()

# ---------------- REALISTIC FRAUD DISTRIBUTION ----------------
# heavily skewed toward low probabilities

fraud_probs = np.random.beta(0.15, 40, n)

# ---------------- RISK LABELS ----------------

risk_labels = []
decisions = []

for p in fraud_probs:

    if p < 0.02:

        risk_labels.append("LOW RISK")
        decisions.append("Approve")

    elif p < 0.08:

        risk_labels.append("MEDIUM RISK")
        decisions.append("Review")

    else:

        risk_labels.append("HIGH RISK")
        decisions.append("Block")

# ---------------- REGIONS ----------------

regions = np.random.choice(
    ["North", "South", "East", "West"],
    size=n
)

# ---------------- AMOUNTS ----------------

amounts = np.random.randint(
    100,
    50000,
    n
)

# ---------------- DATAFRAME ----------------

df = pd.DataFrame({

    "transaction_id": range(1, n + 1),

    "timestamp": timestamps,

    "fraud_probability": fraud_probs,

    "risk_label": risk_labels,

    "decision": decisions,

    "region": regions,

    "amount": amounts
})

# ---------------- SAVE ----------------

df.to_csv(
    "dashboard_data.csv",
    index=False
)

print("dashboard_data.csv created successfully")
