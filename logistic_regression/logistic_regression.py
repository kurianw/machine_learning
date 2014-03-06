import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    def sigmoid(self, z):
        return 1.0/(1 + np.exp(-1*z))

    def correct_estimators(self, learning_rate, samples):
       estimates = self.independent_variables.dot(self.estimators)
       error = estimates - self.dependent_variable
       gradient = self.sigmoid(self.independent_variables.T.dot(error))
       self.estimators -= learning_rate/samples * gradient

    def gradient_descent(self):
       learning_rate = .01
       samples =  self.independent_variables.shape[0]       
       for iteration in xrange(3000):
           self.correct_estimators(learning_rate, samples)    

    def feature_normalization(self, features):
       return (features - self.means)/self.deviations

    def denormalize(self, features):
       return (features + self.means)*self.deviations

    def __init__(self, dataset):
       self.raw = dataset
       independent_variables = dataset[:,:-1]
       self.deviations, self.means = np.std(independent_variables, axis = 0), np.mean(independent_variables, axis = 0)
       self.independent_variables = np.insert(self.feature_normalization(independent_variables), 0, 1, axis = 1)
       self.dependent_variable= self.raw[:,-1][np.newaxis].T
       self.estimators = np.zeros((self.independent_variables.shape[1],1))
       self.gradient_descent() 

    def predict(features):
       normalized = feature_normalization(self, features)
       return sigmoid(self.estimators.dot(np.insert(normalized, 1)))

    def plot(self):
       admitted = self.raw[self.raw[:,2] == 1][:,:-1] 
       not_admitted = self.raw[self.raw[:,2]==0][:,:-1]
       x_range = 30, 100
       y_range = 30, 100
       plt.xlim(x_range)
       plt.ylim(y_range)
       plt.xlabel('Exam 1 score')
       plt.ylabel('Exam 2 score')

       plt.scatter(admitted[:,0], admitted[:,1], marker = '+', color = 'k')
       plt.scatter(not_admitted[:,0], not_admitted[:,1], marker = 'o', color = 'y')
       plt.show()

numerify = lambda x: map(float, x.split(','))
dataset = np.array(map(numerify, open('data/ex2data1.txt', 'r')))

regression = LogisticRegression(dataset)
print regression.estimators 
regression.plot()
