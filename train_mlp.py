#Train MLP
import numpy as np
import matplotlib.pyplot as plt
import pvml
from dict_to_file import *
import sys

feature_function_list = ["color_histogram","edge_direction_histogram",\
						 "cooccurrence_matrix","rgb_cooccurrence_matrix",\
						 "deepfeatures"]
if len(sys.argv) > 1:
	chosen = int(sys.argv[1])
else:
	chosen = 0

feature_function = feature_function_list[chosen]

## LOAD TRAIN DATA
data = np.loadtxt("dataset_processed/train-%s.txt.gz" % (feature_function))
Xtrain = data[:, :-1]
Ytrain = data[:, -1].astype(int) #We return the labels to type int

inputl = Xtrain.shape[1]
nclasses = Ytrain.max() + 1

## DEFINE NETWORK SHAPE: LAYERS AND HYPERPARAMETERS

layers = [inputl, nclasses]

if len(sys.argv) > 2:
	
	epochs = int(sys.argv[2])
	batch_size = int(sys.argv[3])
	lr = float(sys.argv[4])

	#To use hidden layers, set a argv[2] to 1
	if len(sys.argv) > 6:
		hasHidden = (sys.argv[5] is not None and sys.argv[5] == 1)
	else:
		hasHidden = False

	#Extract hidden layers, are passed in format: N1_N2_..._NM
	if hasHidden:
		s = sys.argv[6]
		layers_string = s.split("_")
		hiddenLayers = [int(s) for s in layers_string]		
		layers[1:1] = hiddenLayers
else:
	hasHidden = False
	hiddenLayers = [50]
	if hasHidden:
		layers[1:1] = hiddenLayers

	epochs = 1000
	batch_size = 50
	lr = 0.0015

## ISTANTIATE NETWORK

net = pvml.MLP(layers)

## LOAD VALIDATION DATASET

data = np.loadtxt("dataset_processed/validation-%s.txt.gz" % (feature_function))
Xval = data[:, :-1]
Yval = data[:, -1].astype(int)

## TRAIN NETWORK

for epoch in range(epochs):
	steps = Xtrain.shape[0] // batch_size #For processing all the data in the training steps
	net.train(Xtrain,Ytrain, lr=lr, batch=batch_size, steps=steps)
	predictions, probabilities = net.inference(Xtrain)
	train_acc = (predictions == Ytrain).mean()
	
	predictions, probabilities = net.inference(Xval)
	valid_acc = (predictions == Yval).mean()
	# if epoch % 5 == 0:
	# 	print(epoch ,"Train %f , Test %f" % (train_acc, test_acc))

## SAVE NETWORK

modelname = "mlp-feature_func=%s-layers=%s-lr=%s-epochs=%d-batchsize=%d"\
			 % (feature_function, str(layers), str(lr), epochs, batch_size) 
net.save("models/%s.npz" % modelname) 

## SAVE ACCURACY RESULTS

f = "results/mlp_results.json"
results = dict_from_json(f)
results[modelname] = {"train": train_acc, "validation": valid_acc}
dict_to_json(f, results)