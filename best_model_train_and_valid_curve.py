## PLOT BEST MODEL TRAINING AND VALID CURVE
from dict_to_file import *
import matplotlib.pyplot as plt
import numpy as np


json = dict_from_json("results/best_mlp_train_curves.json") 
train = [t*100 for t in json["train_curve"]]
valid = [t*100 for t in json["valid_curve"]]
nsample = int(json["nsample"])

# print(train)
# print(valid)
# print(nsample)

plt.plot(range(0,nsample*2, 2), train)
plt.plot(range(0,nsample*2, 2), valid)
plt.title("Train and validation versus epochs")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.show()

# print()
iteration_after_which_is_useless = np.argmax(valid)
