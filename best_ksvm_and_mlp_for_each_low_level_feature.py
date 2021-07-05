## Best KSVM for each low level feature
import pandas as pd
from get_df_from_model_name import getParametersFromIndexKSVM
from get_df_from_model_name import getParametersFromIndexMLP

## GET BEST MODEL FOR EACH LOW LEVEL FEATURE KSVM
a = pd.read_json("results/ksvm_results.json").T.reset_index()
details = a.apply(getParametersFromIndexKSVM, axis=1)

cols = details.columns.tolist() + a.columns.tolist()
df = pd.concat([details,a], ignore_index=True, axis=1)
df.columns = cols

m = df.iloc[df.groupby(by="function", axis=0)["validation"].idxmax(),:]
m.to_excel("results/best_ksvm_for_each_category.xlsx")

## GET BEST MODEL FOR EACH LOW LEVEL FEATURE MLP
a = pd.read_json("results/mlp_results.json").T.reset_index()
details = a.apply(getParametersFromIndexMLP, axis=1)

cols = details.columns.tolist() + a.columns.tolist()
df = pd.concat([details,a], ignore_index=True, axis=1)
df.columns = cols
df = df[df["epochs"] != 10000]
# print(df[""].unique())

m = df.iloc[df.groupby(by="feature_func", axis=0)["validation"].idxmax(),:]
m.to_excel("results/best_mlp_for_each_category.xlsx")
