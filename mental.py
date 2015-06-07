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
        self.errors = []
        self.Deltas = []
        self.trained = False

        for (l1, l2) in zip(shape[:-1], shape[1:]):
            self.weights.append(np.random.normal(scale = 0.2, size = (l1+1, l2)))

    def cost(self, X, Y, lamb=0.001):
        X = np.array(X)
        Y = np.array(Y)
        m = float(X.shape[0])

        prediction = self.forward(X)
        costVal = - np.sum(Y * np.log(prediction) + (1 - Y) * np.log(1 - prediction))

        regularization = 0
        for weight in self.weights:
            regularization += np.sum(np.power(weight[1:,:], 2))

        return (costVal + (lamb / 2) * regularization) / m

    def train(self, X, Y, lamb=0.0, alpha=0.1, maxIter=100000):
        # TODO: automatically pick an alpha from [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
        # TODO: automatically pick a lambda?
        # TODO: preprocess training set with feature scaling and etc.

        X = np.array(X)
        Y = np.array(Y)
        m = float(X.shape[0])
        alpha = float(alpha)
        lamb = float(lamb)
        self.trained = True
        self.clear()

        # uncomment to DEBUG
        self.costs = []
        self.dels = []
        self.wegs = []
        # DEBUG END

        for i in range(maxIter):
            # uncomment to DEBUG
            if i % 100 == 0:
                print  i, "- cost:",
                print self.cost(X, Y, lamb),
            self.costs.append(self.cost(X, Y, lamb))
            # DEBUG END

            self.clear()
            self.forward(X)

            # back-propagate:
            # for last layer, just compute error as (last activation - Y)
            self.errors.append(self.a[-1] - Y)
            self.Deltas.append((1.0 / m) * (self.a[-2].T).dot(self.errors[-1]))

            # propagate error through layers
            for layer in reversed(range(1, len(self.weights))):
                lasterror = self.errors[0]
                weightNoBias = self.weights[layer][1:,:]
                newerror = (lasterror.dot(weightNoBias.T)) * self.sgm(self.z[layer - 1], True)
                self.errors.insert(0, newerror)
                self.Deltas.insert(0, ((1.0 / m) * (self.a[layer - 1].T).dot(newerror)))

            # regularize gradients, except for bias row
            for k in range(len(self.Deltas)):
                newDelta = self.Deltas[k]
                newDelta[1:,:] = newDelta[1:,:] + (lamb / m) * self.weights[k][1:,:]
                self.Deltas[k] = newDelta

            # update weights
            for k in range(len(self.weights)):
                self.weights[k] = self.weights[k] - alpha * self.Deltas[k]

            # uncomment to DEBUG
            self.dels.append(self.Deltas[0][1,0])
            self.wegs.append(self.weights[0][1,0])
            if i % 100 == 0:
                print "\tDelta(x1->hidden1):",
                print self.dels[-1],
                print "\tweight(x1->hidden1):",
                print self.wegs[-1]
            # DEBUG END

        # uncomment to DEBUG
        self.clear()
        self.forward(X)

        plt.plot(self.costs)
        plt.show()

        plt.plot(self.dels)
        plt.show()

        plt.plot(self.wegs)
        plt.show()

        plt.scatter(X[:,0], X[:, 1], c=self.a[-1])
        plt.gray()
        plt.show()
        # DEBUG END

        # comment out to DEBUG
        # self.clear()
        # DEBUG END

    def clear(self):
        self.z = []
        self.a = []
        self.errors = []
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
            # a is z activated; output for this layer (with bias if hidden layer)
            self.a.append(zAct)

        return self.a[-1]

    def sgm(self, zee, prime=False):
        if not prime:
            return 1/(1 + np.exp(-zee))
        else:
            sgm = self.sgm(zee)
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

# weight1 = np.array([[-30, 10],[20, -20],[20, -20]])
# weight2 = np.array([[10],[-20],[-20]])
# mn.weights = [weight1, weight2]
