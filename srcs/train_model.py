import os.path
import sys

from MLModel import MLModel


if __name__ == '__main__':
	
	if len(sys.argv) != 2:
		print(f"usage: python3 {os.path.basename(sys.argv[0])} model.csv")
		exit(1)
		
	training_model = MLModel(sys.argv[1])
	# training_model.train()
	training_model.show()
