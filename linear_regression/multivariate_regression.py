import numpy as np
from linear_regression import LinearRegression 

numerify = lambda x: map(float, x.split(','))
dataset = np.array(map(numerify, open('data/ex1data2.txt', 'r')))

print LinearRegression(dataset).estimators 
