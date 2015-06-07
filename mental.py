import numpy as np
import sys

class Neural(object):
    """
    Welcome to mental's NeuralNetwork class.

    shape of network: (features, numHiddenUnits1, numHiddenUnits2, categories)

    each index of self._weights holds the weights for a hidden layer:
           units in hidden layer
    bias: [[unit1, unit2, unit3],
    inp1:  [unit1, unit2, unit3],
    inp2:  [unit1, unit2, unit3]
    inp3:  [unit1, unit2, unit3]]

    X should be of form:
            features of example
    ex1: [[param1, param2, param3],
    ex2:  [param1, param2, param3]]

    Y should be of form (1: belongs to category; 0: doesn't):
    ex1: [[cat1, cat2],
    ex2:  [cat1, cat2]]
    """

    def __init__(self, shape):
        self.shape = shape
        self._weights = []
        self._z = []
        self._a = []
        self._errors = []
        self._Deltas = []
        self.trained = False
        self._initializeWeights()

    def _initializeWeights(self):
        for (l1, l2) in zip(self.shape[:-1], self.shape[1:]):
            self._weights.append(np.random.normal(scale = 0.2, size = (l1+1, l2)))

    def cost(self, X, Y, regParam=0.001):
        X = np.array(X)
        Y = np.array(Y)
        m = float(X.shape[0])

        prediction = self._forward(X)
        costVal = - np.sum(Y * np.log(prediction) + (1 - Y) * np.log(1 - prediction))

        regularization = 0
        for weight in self._weights:
            regularization += np.sum(np.power(weight[1:,:], 2))

        self._clear()

        return (costVal + (regParam / 2) * regularization) / m

    def train(self, X, Y, regParam=0.001, learnRate=0.1, maxIter=100000):
        # TODO: automatically pick an learnRate from [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
        # TODO: automatically pick a regParamda?
        # TODO: preprocess training set with feature scaling and etc.

        X = np.array(X)
        Y = np.array(Y)
        m = float(X.shape[0])
        learnRate = float(learnRate)
        regParam = float(regParam)
        self.trained = True
        self._clear()

        print "training..."

        for i in range(maxIter):
            # print progress bar
            if i % (maxIter / 100) == 0:
                sys.stdout.write(".")
                sys.stdout.flush()

            self._clear()
            self._forward(X)

            # back-propagate:
            # for last layer, just compute error as (last activation - Y)
            self._errors.append(self._a[-1] - Y)
            self._Deltas.append((1.0 / m) * (self._a[-2].T).dot(self._errors[-1]))

            # propagate error through layers
            for layer in reversed(range(1, len(self._weights))):
                lasterror = self._errors[0]
                weightNoBias = self._weights[layer][1:,:]
                newerror = (lasterror.dot(weightNoBias.T)) * self._sgm(self._z[layer - 1], True)
                self._errors.insert(0, newerror)
                self._Deltas.insert(0, ((1.0 / m) * (self._a[layer - 1].T).dot(newerror)))

            # regularize gradients, except for bias row
            for k in range(len(self._Deltas)):
                newDelta = self._Deltas[k]
                newDelta[1:,:] = newDelta[1:,:] + (regParam / m) * self._weights[k][1:,:]
                self._Deltas[k] = newDelta

            # update weights
            for k in range(len(self._weights)):
                self._weights[k] = self._weights[k] - learnRate * self._Deltas[k]

        print "" # creates new line from the progress bar

        self._clear()

    def _clear(self):
        self._z = []
        self._a = []
        self._errors = []
        self._Deltas = []

    def _forward(self, X):
        self._clear()
        X = np.array(X)

        # add bias to input
        inp = np.ones((X.shape[0], X.shape[1] + 1))
        inp[:,1:] = X[:]
        self._a.append(inp)

        for i, W in enumerate(self._weights):
            # a[-1] is last layer's output (with bias)
            # z is the input for this layer
            self._z.append(self._a[-1].dot(W))
            if i == len(self._weights) - 1:
                # just activate output layer
                zAct = self._sgm(self._z[-1])
            else:
                # activate z and add bias
                zAct = np.ones((self._z[-1].shape[0], self._z[-1].shape[1] + 1))
                zAct[:,1:] = self._sgm(self._z[-1])
            # a is z activated; output for this layer (with bias if hidden layer)
            self._a.append(zAct)

        return self._a[-1]

    def predict(self, X, rounding=True):
        X = np.array(X)
        
        # add bias to input
        a = np.ones((X.shape[0], X.shape[1] + 1))
        a[:,1:] = X[:]

        for i, W in enumerate(self._weights):
            # a[-1] is last layer's output (with bias)
            # z is the input for this layer
            z = a.dot(W)
            if i == len(self._weights) - 1:
                # just activate output layer
                zAct = self._sgm(z)
            else:
                # activate z and add bias
                zAct = np.ones((z.shape[0], z.shape[1] + 1))
                zAct[:,1:] = self._sgm(z)
            # a is z activated; output for this layer (with bias if hidden layer)
            a = zAct

        if rounding:
            return np.rint(a)
        else:
            return a

    def _sgm(self, zee, prime=False):
        if not prime:
            return 1/(1 + np.exp(-zee))
        else:
            sgm = self._sgm(zee)
            return sgm * (1 - sgm)

# dummy data for XOR function
training  = [[0.0, 0.0],
             [1.0, 0.0],
             [0.0, 1.0],
             [1.0, 1.0]]

target = [[0.0],
          [1.0],
          [1.0],
          [0.0]]

mn = Neural((2, 2, 1))
mn.train(training, target)
