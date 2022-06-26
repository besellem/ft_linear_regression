import os.path
import sys

from LinearRegressionModel import LinearRegressionModel


if __name__ == '__main__':
	
	if len(sys.argv) != 2:
		print(f"usage: python3 {os.path.basename(sys.argv[0])} model.csv")
		exit(1)
		
	training_model = LinearRegressionModel()
	training_model.train(sys.argv[1])
	# training_model.show(sys.argv[1])
