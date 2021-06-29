#Save file as dicts and recover

import json

def dict_from_json(filename):
	with open(filename, 'r') as fp:
		data = json.load(fp)
		return data

def dict_to_json(filename, dikt):
	if ".json" not in filename:
		filename += ".json"

	with open(filename, 'w') as fp:
		json.dump(dikt, fp)


# a = {"A": {"train": 0, "test": 1}, "B":1}
# f = "results/prova.json"
# dict_to_json(f, a)

# print(dict_from_json(f))
