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
# optional parameters: regularization, learningRate, maxIter
# (coming soon: plotting, activation functions)
```

Make predictions!
```python
network.predict([[0, 1]])   # [[1]]
```
