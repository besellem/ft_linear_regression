import os.path
import sys

from LinearRegressionModel import LinearRegressionModel


if __name__ == '__main__':
	
	if len(sys.argv) > 1:
		print(f"usage: python3 {os.path.basename(sys.argv[0])}")
		exit(1)

	s = float(input('Search : '))

	LR = LinearRegressionModel()
	LR.test(sys.argv[1])
