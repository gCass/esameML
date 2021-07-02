## CONVERT DF INTO IMAGE AFTER GETTING CHARACTERISTICS FROM NAME
import pandas as pd
df = pd.read_csv("results/results_df.csv")

ksvm = df.iloc[0:5,:]
mlp = df.iloc[5:,:]

def getParametersFromIndexKSVM(row):
	s = row["index"]
	l = s.split("_")
	model = l[0]
	kfun = l[1].split("=")[1]
	lambd = l[3].lstrip("=")
	kparam = l[4].split("=")[1]
	lr = l[5].split("=")[1]
	#Get the function
	find_double_underscore_position = s.find("__")
	function = s[find_double_underscore_position+2:]
	return pd.Series({"model": model, "kfun": kfun, "lambda": lambd, "kparam": kparam, "lr": 0.1, "function": function})

d = ksvm.apply(getParametersFromIndexKSVM,axis=1)
prefixdf_ksvm = pd.DataFrame(d)
print(prefixdf_ksvm)

final_ksvm = pd.concat([prefixdf_ksvm, ksvm],axis=1)
print(final_ksvm)

final_ksvm.to_csv("results/top_ksvm_results_name_translated.csv")


def getParametersFromIndexMLP(row):
	s = row["index"]
	l = s.split("-")
	print(l)
	d = {}
	d["model"] = "mlp"
	for element in l[1:]:
		attr_name = element.split("=")[0]
		attr_val = element.split("=")[1]
		d[attr_name] = attr_val

	return pd.Series(d) 

d = mlp.apply(getParametersFromIndexMLP,axis=1)
prefixdf_mlp = pd.DataFrame(d)
print(prefixdf_ksvm)

final_mlp = pd.concat([prefixdf_mlp, mlp],axis=1)
final_mlp.to_csv("results/top_mlp_results_name_translated.csv")
