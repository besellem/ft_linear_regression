import matplotlib.pyplot as plt
import pandas as pd


class LinearRegressionModel:
	
	def __init__(self):
		self.__theta0 = 0
		self.__theta1 = 0
		self.__x = []
		self.__y = []

	def train(self, csv_file: str, out_file: str = ".trained_model"):
		dataframe = pd.read_csv(csv_file)
		
		if len(dataframe.columns) != 2:
			# raise 'Error: Can only process 2 columns CSVs'
			print('Error: Can only process 2 columns CSVs')
			return
		
		data = dataframe.values.reshape(-1, 2)
		for line in data:
			self.__x.append(line[0])
			self.__y.append(line[1])
		
		print('x:', self.__x)
		print('y:', self.__y)
	
		theta0_tmp, theta1_tmp = (0, 0)
		ratio = .01
		size = len(self.__x)
		reps = 2
		
		for _ in range(reps):
			for i in range(size):
				theta0_tmp += ratio * ((1 / size) * self.estimate(self.__x[i]) - self.__y[i])
				theta1_tmp += ratio * (theta0_tmp * self.__x[i])
		
			self.__theta0 = theta0_tmp
			self.__theta1 = theta1_tmp

		print('theta0:', self.__theta0)
		print('theta1:', self.__theta1)

		# theta0_tmp = ratioDApprentissage * sum (estimate(km[i]) − price[i])
		# theta1_tmp = ratioDApprentissage * sum (estimate(km[i]) − price[i]) * km[i]
	
	def test(self, model_file: str):
		print(f'using {model_file}...')
	
	def estimate(self, km: int):
		return (self.__theta1 * km) + self.__theta0
	
	def show(self, csv_file: str, resolve=True):
		dataframe = pd.read_csv(csv_file)
		
		# if resolve:
		# 	self.use()
		# 	plt.plot([1000, 250000], [8000, 1000], color='b', alpha=.7)
		
		headers = dataframe.columns
		plt.scatter(self.__x, self.__y, color='g', alpha=.7)
		plt.xlabel(headers[0])
		plt.ylabel(headers[1])
		plt.title(self.__module__)
		plt.legend()
		plt.grid()
		plt.show()
