#Feature extraction runner
import os
for chosen_number in range(0, 5):
	os.system('python3 feature_extraction.py %d' % chosen_number)
