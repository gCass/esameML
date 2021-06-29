## MLP RUNNER CHOOSING HYPERPARAMETER WITH GREED SEARCH
import os


epochs_array = [100]#, 1000, 10000]
batch_sizes = [10,20,30,40,50]
lrs = [0.001, 0.002, 0.003, 0.01, 0.02, 0.03, 0.1 , 0.2, 0.3]

# LINEAR
for chosen_number in range(0, 4):
	for epochs in epochs_array:
		for batch_size in batch_sizes:
			for lr in lrs:
				print("Doing ",chosen_number, epochs, batch_size, lr)
				lambda_ = random.choice(lambdas)
				gamma = random.choice(gammas)
				lr = random.choice(lrs)
				os.system('python3 train_mlp.py %d %d %d %f %d' % (chosen_number, epochs, batch_size, lr, 0))

# NON LINEAR