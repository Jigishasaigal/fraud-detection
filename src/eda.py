import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../data/creditcard.csv")

print("Dataset shape:", df.shape)
print("Class distribution:")
print(df["Class"].value_counts(normalize=True))

sns.countplot(x="Class", data=df)
plt.title("Fraud vs Non-Fraud Distribution")
plt.show()
