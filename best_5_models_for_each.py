#get best top 5 models for both ksvm and mlp
import pandas as pd

def getTop5Rows(df, column):
	top_df = df.sort_values(by=[column], ignore_index=False, ascending=False)
	return top_df.iloc[0:5,:]

ksvm_df = pd.read_json("results/ksvm_results.json").T
top_ksvm = getTop5Rows(ksvm_df, "validation")

mlp_df = pd.read_json("results/mlp_results.json").T
top_mlp = getTop5Rows(mlp_df, "validation")

# print(top_mlp)
# print(top_ksvm)

total = pd.concat([top_ksvm, top_mlp])
total.reset_index(level=0, inplace=True)
total.to_csv("results/results_df.csv",index=True)


