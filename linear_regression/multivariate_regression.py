import numpy as np
from linear_regression import LinearRegression 

numerify = lambda x: map(float, x.split(','))
dataset = np.array(map(numerify, open('data/ex1data2.txt', 'r')))

lin = LinearRegression(dataset)
print lin.estimators
print lin.predict([1650,3])
