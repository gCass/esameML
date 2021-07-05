#plots
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("results/results_df.csv")
print(df.columns)
sns.set_theme(style="white", context="talk")
# rs = np.random.RandomState(8)

df["train"] = df["train"] * 100
df["validation"] = df["validation"] * 100

# Set up the matplotlib figure
# f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 5), sharex=True)

# Generate some sequential data
# x = total.index
# y1 = 
# print(a)
df.sort_values(by=["validation"],inplace=True, ascending=False)
df[["train","validation"]].plot.barh()
plt.title("Train and validation accuracy")
plt.ylabel("Model index")
plt.xlabel("Accuracy")
plt.tight_layout()	
plt.savefig("plots/train_and_validation_accuracy")

