#Feature extraction for the exam
import pvml
import numpy as np 
import matplotlib.pyplot as plt
import os
import image_features
import sys

## Define a fuction for extracting a features as output of a CNN pretrained

def extract_neural_features(im, net, previous_layers = 0, different_level_deep = False):
	im = im[None, :, :, :]
	activations = net.forward(im)
	features  = activations[-3-previous_layers]
	if not different_level_deep:
		features = features.reshape(-1)
	else:
		#Per ogni channel, faccio la media lungo gli assi 0 e 1 che sono riga e colonna
		#dell'immagine 
		features = np.average(features[0,:,:,:], axis=(0,1))

	return features

classes = os.listdir("dataset/train/")
classes.sort()
print(classes)

feature_function_list = ["color_histogram","edge_direction_histogram", \
						 "cooccurrence_matrix","rgb_cooccurrence_matrix",\
						 "deepfeatures"]
if len(sys.argv) > 1:
	chosen = int(sys.argv[1])
else:
	chosen = 0
feature_function = feature_function_list[chosen] 

def process_directory(path):
	net = pvml.CNN.load("cake-classification/pvmlnet.npz")
	all_features = []
	all_labels = []
	class_labels_counter = 0
	#Walk through all folders and convert all into numpy array
	for class_ in classes:
		#For each class we can build the path by putting together the root
		class_path = path + "/" + class_
		image_files = os.listdir(class_path)
		#Now for each image in the folder we load image, extract feature
		for imagename in image_files:
			#For reading the image we can use an utility function of matplotlib
			image = plt.imread(class_path + "/" + imagename)			
			#The image will be loaded with an encoding of 1 byte per any channel integer number. 
			#We prefer to work with floating point number between [0,1]. 
			#To do this, divide by 255
			image = image / 255
			#Extract different features by choice
			if feature_function == "color_histogram":
				features = image_features.color_histogram(image)
			if feature_function == "edge_direction_histogram":
				features = image_features.edge_direction_histogram(image)
			elif feature_function == "cooccurrence_matrix":				
				features = image_features.cooccurrence_matrix(image)
			elif feature_function == "rgb_cooccurrence_matrix":
				features = image_features.rgb_cooccurrence_matrix(image)
			elif feature_function == "deepfeatures":
				features = extract_neural_features(image, net) 	

			# reshape to get a row vector
			features = features.reshape(-1) 
			#Give a look at this features 
			# plt.bar(np.arange(features.shape[0]), features)
			# plt.show()
			#At this point we have to collect all our feature vectors
			all_features.append(features)
			all_labels.append(class_labels_counter)
		#Increment the counter
		class_labels_counter += 1
	#Now we can stack the list of features along dimension 0 (dimension of rows), to
	#get a single matrix 
	X = np.stack(all_features,0)
	Y = np.array(all_labels)
	return X, Y


X, Y = process_directory("dataset/test")
data = np.concatenate([X, Y[:,None]],1)
print(feature_function)
p = "dataset_processed/test-%s.txt.gz" % feature_function
print(p)
np.savetxt(p, data)

X, Y = process_directory("dataset/validation")
print(X.shape, Y.shape)
data = np.concatenate([X, Y[:,None]],1)
p = "dataset_processed/validation-%s.txt.gz" % feature_function
print(p)
np.savetxt(p, data)

X, Y = process_directory("dataset/train")
print(X.shape, Y.shape)
data = np.concatenate([X, Y[:,None]],1)
p = "dataset_processed/train-%s.txt.gz" % feature_function
print(p)
np.savetxt(p, data)
