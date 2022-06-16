import os.path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class MLModel:
	
	def __init__(self, in_file: str, model: str = "linear-regression"):
		self.__model = model
		self.__in_file = in_file
		self.__in_df = pd.read_csv(in_file)
	
	def train(self, out_file: str = ".trained_model"):
		if self.__model == "linear-regression":
			self.__train_linear_regression(out_file)
	
	def use(self):
		if self.__model == "linear-regression":
			self.__linear_regression()
	
	def __train_linear_regression(self, out_file: str):
		print(f'training {out_file}...')
	
	def __linear_regression(self):
		print(f'using {self.__in_file}...')


def print_usage():
	print(f"usage: python3 {os.path.basename(sys.argv[0])} model.csv")


if __name__ == '__main__':
	
	if len(sys.argv) != 2:
		print_usage()
		exit(1)
	
	training_model = MLModel(sys.argv[1])
	training_model.use()
	

# file = pd.read_csv(sys.argv[1])
# cols = file.columns
# n = np.array(file)
#
# l1 = [0, 1, 2, 3, 4, 5, 6, 7, 9, -10, 2]
# l2 = [20, -1, 32, -12, 6, 0, -2, 4, -12, 0, 2]
# data = list(map(lambda x, y: (x, y), l1, l2))
#
# print(data)

# plt.title('linear_regression')
# # plt.scatter(1, 0)
# plt.xlabel('price')
# plt.ylabel('km')
# plt.show()
