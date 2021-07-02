import pvml
import numpy as np 
import matplotlib.pyplot as plt
import os
from train_ksvm import OVO_KSVM


def getTestAccuracyAndConfusionMatrix(model, modelname, Xtest, Ytest, nclasses, classes):
	labels, probs = model.inference(Xtest)

	acc = (Ytest == labels).mean()
	print(acc)

	cm = np.zeros((nclasses, nclasses))

	for i in range(Xtest.shape[0]):
			cm[Ytest[i],labels[i]] += 1

	cm = cm / cm.sum(1, keepdims=True)

	print(" " * nclasses, end="")
	for j in range(nclasses):
		print(classes[j][:3], end=" ")
	print()

	for i in range(nclasses):
		print("%9s" % classes[i], end=" ")
		for j in range(nclasses):
			#Print a tabulation and not a new line character after doing the print
			print("%.3f" % cm[i,j], end="\t")
		print()

	plt.imshow(cm)
	for i in range(nclasses):
		for j in range(nclasses):
			#When you place the text you don't put coordinates as i j but as j i.
			plt.text(j,i, int(100 * cm[i,j]), color="orange", size=20)

	plt.xticks(range(nclasses),classes, rotation=90)
	plt.yticks(range(nclasses),classes) 
	plt.title("Confusion matrix")
	plt.tight_layout()
	plt.show()


#Build confusion matrix a see if there is any sense in the confusion to see which kind of cakes
#are confused

feature_function_list = ["color_histogram","edge_direction_histogram", "cooccurrence_matrix","rgb_cooccurrence_matrix","deepfeatures"]
chosen = -1
feature_function = feature_function_list[chosen] 

#Load dataset
data = np.loadtxt("dataset_processed/test-%s.txt.gz" % feature_function)
Xtest = data[:, :-1]
Ytest = data[:, -1].astype(int)

nclasses = Ytest.max() + 1
classes = os.listdir("dataset/test/")
#Sort the name of the classes
classes.sort()

modelname = "mlp-feature_func=deepfeatures-layers=[1024, 3]-lr=0.01-epochs=100-batchsize=10.npz"



# RETRIEVE CHE CORRECT MODEL

if ".npz" in modelname:
	modelname_stripped = modelname.rstrip(".npz")
	model = pvml.MLP.load("models/%s" % modelname)
elif ".pkl" in modelname:
	model = OVO_KSVM().load(modelname)

# END RETRIEVE



#Load network

getTestAccuracyAndConfusionMatrix(net, modelname, Xtest, Ytest, nclasses, classes)
