import numpy as np
from linear_regression import LinearRegression 
import matplotlib.pyplot as plt

class UnivariateRegression(LinearRegression):

    def plot(self):
        independent_variable = self.independent_variables[:,1]
        plt.scatter(independent_variable, self.dependent_variable, marker = 'x', color = 'r') 
        x_range = independent_variable.min(), independent_variable.max()
        print x_range
        y_range = self.dependent_variable.min(), self.dependent_variable.max()
        print y_range
        y_values = map(lambda x: self.predict([x]), x_range)
        plt.xlim(x_range[0] - 1, x_range[1] + 1)
        plt.ylim(y_range[0] - 1, y_range[1] + 1)
        plt.plot(x_range, y_values)
        plt.show()


numerify = lambda x: map(float, x.split(','))
dataset = np.array(map(numerify, open('data/ex1data1.txt', 'r')))

regression = UnivariateRegression(dataset)
print regression.estimators 
regression.plot()
