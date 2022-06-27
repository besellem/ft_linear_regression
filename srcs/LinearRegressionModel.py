import matplotlib.pyplot as plt
import pandas as pd


THETAS_FILE = ".trained"


class LinearRegressionModel:

	def __init__(self, csv_data: str):
		file = pd.read_csv(csv_data)
		if len(file.columns) != 2:
			raise IOError('Error: Can only process 2 columns CSVs')
		data = file.values.reshape(-1, 2)

		self.headers = file.columns
		self.x, self.y = data[:, 0], data[:, 1]
		self.th0, self.th1 = self.__parse_thetas()

	def train(self, epoch=100, alpha=1):
		for _ in range(epoch):
			self.th0, self.th1 = self.__process_thetas(alpha)

	def show(self):
		line_x = [self.x.min(), self.x.max()]
		line_y = [self.estimate(x) for x in [0, 1]]
		
		plt.scatter(self.x, self.y, color='green', alpha=.5, label="data")
		plt.plot(line_x, line_y, color='black', alpha=.7, label="model")
		plt.xlabel(self.headers[0])
		plt.ylabel(self.headers[1])
		plt.title(self.__module__)
		plt.legend()
		plt.grid()
		plt.show()

	def predict(self, x):
		val = self.__normalize_x(x)
		return self.estimate(val)

	def estimate(self, x):
		return (self.th1 * x) + self.th0

	# cost function (or MSE)
	def cost(self):
		normalized_x = self.__normalize_x(self.x)
		return sum([(self.estimate(x) - y) ** 2 for x, y in zip(normalized_x, self.y)]) / (len(self.x))

	def save(self, th0=None, th1=None):
		if (th0 and not th1) or (not th0 and th1):
			raise Exception('Cannot save data, theta0 or theta1 missing')

		f = open(THETAS_FILE, "w")
		f.write(f'{th0 if th0 else self.th0}:{th1 if th1 else self.th1}')
		f.close()

	def __normalize_x(self, x):
		return (x - self.x.min()) / (self.x.max() - self.x.min())

	def __process_thetas(self, alpha):
		normalized_x = self.__normalize_x(self.x)
		theta0_tmp_list = []
		theta1_tmp_list = []

		for x, y in zip(normalized_x, self.y):
			theta0_tmp_list.append(self.estimate(x) - y)
			theta1_tmp_list.append((self.estimate(x) - y) * x)
		
		x_size = len(normalized_x)
		theta0 = self.th0 - alpha * (sum(theta0_tmp_list) / x_size)
		theta1 = self.th1 - alpha * (sum(theta1_tmp_list) / x_size)

		return theta0, theta1

	def __parse_thetas(self):
		try:
			f = open(THETAS_FILE, 'r')
			raw = f.read()
			f.close()
			data = raw.split(':')
			th0 = float(data[0])
			th1 = float(data[1])
			return th0, th1
		except IOError:
			return 0, 0
