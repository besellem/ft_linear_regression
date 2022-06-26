import sys

from LinearRegressionModel import *


if __name__ == '__main__':
	
	if len(sys.argv) != 2:
		print('usage: python3 ft_linear_regression.py .model')
		exit(1)

	s = int(input('Search : '))

	LR = LinearRegressionModel()
	LR.test(sys.argv[1])
