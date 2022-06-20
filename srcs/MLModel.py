import matplotlib.pyplot as plt
import pandas as pd


MODELS = ["linear-regression"]


class MLModel:
	
	def __init__(self, in_file: str, model: str = "linear-regression"):
		
		if model not in MODELS:
			print(f'Error: model "{model}" does not exist')
			print('Existing models:')
			print('\t', *MODELS)
			exit(1)
		
		self.__model = model
		self.__in_file = in_file
		self.__dataframe = pd.read_csv(in_file)
		
		if len(self.__dataframe.columns) != 2:
			print('Error: Can only process 2 columns CSVs')
			exit(1)
		
		self.__x = []
		self.__y = []
		self.__theta0 = 0
		self.__theta1 = 0
		
		data = self.__dataframe.values.reshape(-1, 2)
		for line in data:
			self.__x.append(line[0])
			self.__y.append(line[1])
	
	def train(self, out_file: str = ".trained_model"):
		if self.__model == "linear-regression":
			self.__train_linear_regression(out_file)
	
	def use(self):
		if self.__model == "linear-regression":
			self.__use_linear_regression()
	
	def __train_linear_regression(self, out_file: str):
		print(f'training {out_file}...')
		print((self.__x, self.__y))

	def __use_linear_regression(self):
		print(f'using {self.__in_file}...')

	def show(self, resolve=True):
		
		if resolve:
			self.use()
			plt.plot([1000, 250000], [8000, 1000], color='b', alpha=.7)
		
		headers = self.__dataframe.columns
		plt.scatter(self.__x, self.__y, color='g', alpha=.7)
		plt.xlabel(headers[0])
		plt.ylabel(headers[1])
		plt.title(self.__model + ' model')
		plt.legend()
		plt.show()
