def gradient_descent(gradient, estimators, args=(), learning_rate = .01, iterations = 3000 ):
    for iteration in xrange(iterations):
        estimators -= learning_rate * gradient(estimators, *args)
    return estimators
