import numpy as np

class LinearRegression:
    def correct_estimators(self, learning_rate, samples):
        estimates = self.independent_variables.dot(self.estimators)
        error = estimates - self.dependent_variable
        gradient = self.independent_variables.T.dot(error)
        self.estimators -= learning_rate/samples * gradient
 
    def gradient_descent(self):
        learning_rate = .01
        samples =  self.independent_variables.shape[0]       
        for iteration in xrange(3000):
            self.correct_estimators(learning_rate, samples)

    def feature_normalization(self, features):
        return (features - self.means)/self.deviations
 
    def __init__(self, dataset):
        independent_variables = dataset[:,:-1] 
        self.deviations, self.means = np.std(independent_variables, axis = 0), np.mean(independent_variables, axis = 0)
        self.independent_variables = np.insert(self.feature_normalization(independent_variables), 0, 1, axis = 1)
        self.dependent_variable= dataset[:,-1][np.newaxis].T
        self.estimators = np.zeros((self.independent_variables.shape[1],1))
        self.gradient_descent() 
    
    def predict(self,features):
        normalized = self.feature_normalization(features)
        return np.insert(features, 0, 1).dot(self.estimators)
