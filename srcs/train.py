import os.path
import sys

from LinearRegressionModel import LinearRegressionModel


if __name__ == '__main__':
	
	if len(sys.argv) > 2:
		print(f"usage: python3 {os.path.basename(sys.argv[0])} [model.csv]")
		exit(1)
	
	try:
		if (len(sys.argv) == 2):
			lr = LinearRegressionModel(sys.argv[1])
		else:
			lr = LinearRegressionModel('./data/data.csv')
	except IOError as e:
		print(e, file=sys.stderr)
		exit(1)
	
	lr.train(epoch=1000)
	print([lr.th0, lr.th1])
	lr.save()
	lr.show()
