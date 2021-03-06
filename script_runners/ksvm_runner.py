## KSVM RUNNER CHOOSING HYPERPARAMETER RANDOMLY
import os
import random

#SET A SEED SUCH THAT YOU CAN REPEAT EXPERIMENTS
random.seed(a=7777777)

lambdas = [0.001, 0.002, 0.003, 0.01, 0.02, 0.03, 0.1 , 0.2, 0.3]
gammas = [0.001, 0.002, 0.003, 0.01, 0.02, 0.03, 0.1 , 0.2, 0.3]
lrs = [0.001, 0.002, 0.003, 0.01, 0.02, 0.03, 0.1 , 0.2, 0.3]

FIXED = False
start = 0
if FIXED:
	start = 4

for chosen_number in range(start, 5):
	## EXTRACT RANDOMLY 3 TIMES FOR EACH CHOSEN NUMBER A SET OF HYPERPARAMETERS
	for i in range(0,10):
		print("Doing ",chosen_number, i)
		lambda_ = random.choice(lambdas)
		gamma = random.choice(gammas)
		lr = random.choice(lrs)
		os.system('python3 train_ksvm.py %d %f %f %f' % (chosen_number, lambda_, gamma, lr))
