import numpy as np
from gradient_descent import gradient_descent

class LinearRegression:
    def gradient(self, theta, X, y, m):
        estimates = X.dot(theta)
        error = estimates - y 
        return 1.0/m * X.T.dot(error)

    def cost_function(self, theta, X, y, m):
        estimates = X.dot(theta)
        error = estimats - y
        return 1.0/(2*m)*error.T.dot(error)

    def __init__(self, dataset):
        independent_variables = dataset[:,:-1] 
        self.independent_variables = np.insert(independent_variables, 0, 1, axis = 1)
        self.dependent_variable= dataset[:,-1][np.newaxis].T
        estimators = np.zeros((self.independent_variables.shape[1],1))
        self.estimators = gradient_descent(self.gradient, estimators, args = (self.independent_variables, self.dependent_variable, self.independent_variables.shape[0])) 
    
    def predict(self,features):
        return np.insert(features, 0, 1).dot(self.estimators)[0]
