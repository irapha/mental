# mental
A python framework for Neural Networks on-the-go

### How to use it
Create a Neural Network
```python
from mental import Neural

network = Neural((2, 2, 1))   # two inputs units, two hidden units, two outputs
```

Train it (rows of examples, columns of features)
```python
trainingSet  = [[0, 0],
                [1, 0],
                [0, 1],
                [1, 1]]

targetValues = [[0],
                [1],
                [1],
                [0]]

network.train(trainingSet, targetValues)
# optional parameters: regularization, learningRate, maxIter, testing
# (coming soon: more plotting, more activation functions)
```

Make predictions!
```python
network.predict([[0, 1]])   # [[1]]
```

### More options!
Print more information and plot cost over training iterations
```python
testing = True   # prints and plots
network.train(trainingSet, targetValues, 0.001, 0.01, 100000, testing)
```
![Cost vs. Iterations](/imgs/costvsiter.png?raw=true =200x "Cost vs. Training Iterations")

Plot cost of training set (Jtrain) and cost of validation set (Jcv) as a function of the training set size.
```python
import matplotlib.pyplot as plt
trainCosts = []
cvCosts = []

for i in range(2, trainFeats.shape[0], 6):
    net = Neural((13, 30, 20, 10, 3))
    net.train(trainFeats[:i], trainTargs[:i], 0.01, 0.01, 50000)
    trainCosts.append(net.cost(trainFeats[:i], trainTargs[:i], 0.01))
    cvCosts.append(net.cost(cvFeats, cvTargs, 0.01))

plt.plot(trainCosts)
plt.plot(cvCosts)
plt.show()
```
// screenshot coming! (it's training.. haha)
