import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin

class LogisticRegression:
    def sigmoid(self, z):
        return 1.0/(1 + np.exp(-1*z))

    def correct_estimators(self, learning_rate, samples):
       estimates = self.independent_variables.dot(self.estimators)
       error = self.sigmoid(estimates) - self.dependent_variable
       gradient = self.independent_variables.T.dot(error)
       self.estimators -= learning_rate/samples * gradient

    def gradient_descent(self):
       learning_rate = .01
       samples =  self.independent_variables.shape[0]       
       for iteration in xrange(3000):
           self.correct_estimators(learning_rate, samples)    

    def cost_function(self,theta,X,y):
        y_vector = y[:,-1]
        predictions = lambda selector: self.sigmoid(X[y_vector==selector].dot(theta))
        false_positive_costs = -1*np.log(predictions(1))
        false_negative_costs = -1*np.log(1-predictions(0))
        return 1.0/len(y)*(false_positive_costs.sum() + false_negative_costs.sum())

    def feature_normalization(self, features):
       return (features - self.means)/self.deviations

    def __init__(self, dataset):
       self.dataset = dataset
       independent_variables = dataset[:,:-1]
       self.deviations, self.means = np.std(independent_variables, axis = 0), np.mean(independent_variables, axis = 0)
       self.independent_variables = np.insert(independent_variables, 0, 1, axis = 1)
       self.dependent_variable= self.dataset[:,-1][np.newaxis].T
       estimators = np.zeros((self.independent_variables.shape[1],1))
       self.estimators = fmin(self.cost_function, estimators, args = (self.independent_variables, self.dependent_variable))

    def predict(self, features):
        return self.sigmoid(np.insert(features, 0, 1).dot(self.estimators))

    def plot(self):
       partition = lambda selector: self.dataset[self.dataset[:,2] == selector][:,:-1]
       admitted = partition(1) 
       not_admitted = partition(0) 
       x_range = 30, 100
       y_range = 30, 100
       y_values = map(self.predict, zip(x_range, y_range))
       plt.xlim(x_range)
       plt.ylim(y_range)
       plt.xlabel('Exam 1 score')
       plt.ylabel('Exam 2 score')

       plt.scatter(admitted[:,0], admitted[:,1], marker = '+', color = 'k')
       plt.scatter(not_admitted[:,0], not_admitted[:,1], marker = 'o', color = 'y')
       plt.plot(x_range,y_values)
       plt.show()

dataset = np.loadtxt('data/ex2data1.txt', delimiter=',')

regression = LogisticRegression(dataset)
print regression.predict([45,85])
