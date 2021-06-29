#Train ksvm
import numpy as np
import matplotlib.pyplot as plt
import pvml
from dict_to_file import *
import sys
import pickle

class OVO_KSVM:
	def __init__(self, lambda_=0.001, kfun="rbf", kparam=0.01, lr=1e-3):
		self.kfun = kfun
		self.lambda_ = lambda_
		self.kparam = kparam
		self.lr = lr
		self.alpha = None
		self.b = None

	def train(self,X,Y):
		alpha, b = pvml.one_vs_one_ksvm_train(X, Y, self.kfun, self.kparam, self.lambda_, lr=self.lr, steps=1000,
                          init_alpha=None, init_b=None)
		self.alpha = alpha
		self.b = b
		self.Xtrain = X

	def inference(self,X,Y):
		 return pvml.one_vs_one_ksvm_inference(X, self.Xtrain, self.alpha, self.b, self.kfun, self.kparam)

	def save(self,path):
		with open(path, 'wb') as output:
			pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

	def load(self, path):
		with open(path, 'rb') as input:
			tmp_dict = pickle.load(input)

		self.__dict__.clear()
		self.__dict__.update(tmp_dict.__dict__) 

		# print(self.Xtrain)

	def getAttributeString(self):
		dikt = self.__dict__
		s = ""
		for key in dikt:
			if key != "alpha" and key != "b" and key != "Xtrain":
				s += str(key) + "=" + str(dikt[key]) + "_"
		return s 


# print("ciao")
feature_function_list = ["color_histogram","edge_direction_histogram", "cooccurrence_matrix","rgb_cooccurrence_matrix"]
if len(sys.argv) > 1:
	chosen = int(sys.argv[1])
else:
	chosen = 0



## LOAD TRAIN DATA
feature_function = feature_function_list[chosen]
data = np.loadtxt("dataset_processed/train-%s.txt.gz" % (feature_function))
Xtrain = data[:, :-1]
Ytrain = data[:, -1].astype(int) #We return the labels to type int


# CREATE DEFAULT MODEL OR WITH PARAMS
if len(sys.argv) >= 4:
	lambda_ = float(sys.argv[2])
	kparam= float(sys.argv[3])
	lr = float(sys.argv[4])
	model = OVO_KSVM(lambda_=lambda_, kparam=kparam, lr=lr)
else:
	model = OVO_KSVM()

## TRAIN THE MODEL
model.train(Xtrain, Ytrain)

## GET TRAINING ACCURACY
pred_label, _ = model.inference(Xtrain, Ytrain)
train_acc = (Ytrain == pred_label).mean()

## SAVE THE MODEL
s = model.getAttributeString()
model.save("models/ksvm_"+s+".pkl")

modelname = "ksvm_"+s+"_"+feature_function

##LOAD VALIDATION DATA
data = np.loadtxt("dataset_processed/validation-%s.txt.gz" % (feature_function))
Xval = data[:, :-1]
Yval = data[:, -1].astype(int)

##GET VALIDATION ACCURACY
pred_label, _ = model.inference(Xval, Yval)
validation_acc = (Yval == pred_label).mean()

## SAVE ACCURACY RESULTS
f = "results/ksvm_results.json"
results = dict_from_json(f)
results[modelname] = {"train": train_acc, "validation": validation_acc}
dict_to_json(f, results)