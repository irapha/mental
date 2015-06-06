import numpy as np
import matplotlib.pyplot as plt

class Neural(object):
    """
    Welcome to mental's NeuralNetwork class.

    shape of network: (features, numHiddenUnits1, numHiddenUnits2, categories)

    each index of self.weights holds the weights for a hidden layer:
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
        self.weights = []
        self.z = []
        self.a = []
        self.deltas = []
        self.Deltas = []
        self.trained = False

        # uncomment to DEBUG
        self.costs = []

        for (l1, l2) in zip(shape[:-1], shape[1:]):
            self.weights.append(np.random.normal(scale = 0.2, size = (l1+1, l2)))

    def cost(self, X, Y, lamb=1):
        X = np.array(X)
        Y = np.array(Y)

        # number of examples:
        m = X.shape[0]

        prediction = self.forward(X)
        costVal = - np.sum(Y * np.log(prediction) + (1 - Y) * np.log(1 - prediction))

        regularization = 0
        for weight in self.weights:
            regularization += np.sum(np.power(weight, 2))

        return (costVal + (lamb / 2) * regularization) / m

    def train(self, X, Y, lamb=0.01, alpha=0.03, maxIter=100000):
        # TODO: automatically pick an alpha from [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
        # TODO: prepare training first by scaling vars and etc

        X = np.array(X)
        Y = np.array(Y)
        m = float(X.shape[0])
        self.trained = True
        self.clear()

        # uncomment to DEBUG
        self.costs = []

        for i in range(maxIter):

            if i % 100 == 0:
                print self.cost(X, Y, lamb)
            self.costs.append(self.cost(X, Y, lamb))

            self.clear()
            self.forward(X)

            # back-propagate:
            # for last layer, just compute delta as activation - Y
            self.deltas.append(self.a[-1] - Y)
            self.Deltas.append((self.a[-2].T).dot(self.deltas[-1]))

            for layerIndex in reversed(range(1, len(self.weights))):
                lastdelta = self.deltas[0]
                # calculate newdelta
                newdelta = (lastdelta * self.sgm(self.z[layerIndex], True)).dot(self.weights[layerIndex].T)
                # remove bias from newdelta
                newdelta = newdelta[:,1:]
                self.deltas.insert(0, newdelta)
                self.Deltas.insert(0, (self.a[layerIndex - 1].T).dot(newdelta))

            for k in range(len(self.Deltas)):
                newDelta = (1.0 / m) * self.Deltas[k]
                newDelta[1:,:] = newDelta[1:,:] + (lamb / m) * self.weights[k][1:,:]
                self.Deltas[k] = newDelta

            for k in range(len(self.weights)):
                self.weights[k] = self.weights[k] - float(alpha) * self.Deltas[k]

        # uncomment to DEBUG
        self.clear()
        self.forward(X)

        plt.plot(self.costs)
        plt.show()

        plt.scatter(X[:,0], X[:, 1], c=self.a[-1])
        plt.gray()
        plt.show()

        # comment to DEBUG
        # self.clear()

    def clear(self):
        self.z = []
        self.a = []
        self.deltas = []
        self.Deltas = []

    def forward(self, X):
        self.clear()
        X = np.array(X)

        # add bias to input
        inp = np.ones((X.shape[0], X.shape[1] + 1))
        inp[:,1:] = X[:]
        self.a.append(inp)

        for i, W in enumerate(self.weights):
            # a[-1] is last layer's output (with bias)
            # z is the input for this layer
            self.z.append(self.a[-1].dot(W))
            if i == len(self.weights) - 1:
                # just activate output layer
                zAct = self.sgm(self.z[-1])
            else:
                # activate z and add bias
                zAct = np.ones((self.z[-1].shape[0], self.z[-1].shape[1] + 1))
                zAct[:,1:] = self.sgm(self.z[-1])
            # a is z activated; output for this layer (with bias)
            self.a.append(zAct)

        return self.a[-1]

    def sgm(self, zee, prime=False):
        if not prime:
            return 1/(1 + np.exp(-zee))
        else:
            sgm = self.sgm(zee)
            return sgm * (1 - sgm)

# dummy data for XOR function
training  = [[0, 0],
             [1, 0],
             [0, 1],
             [1, 1]]

target = [[1],
          [0],
          [0],
          [1]]

mn = Neural((2, 2, 1))
mn.train(training, target)

# weight1 = np.array([[-30, 10],[20, -20],[20, -20]])
# weight2 = np.array([[10],[-20],[-20]])
# mn.weights = [weight1, weight2]
