#Feature extraction runner
import os
for chosen_number in range(0, 4):
	os.system('python3 feature_extraction.py %d' % chosen_number)
