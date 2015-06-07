import sys
sys.path.append("..")
from mental.mental import Neural
import numpy as np

# # # DEMO 1 - XOR
# drunken = Neural((2, 2, 1))
# drunken.train([[0,0],[0,1],[1,0],[1,1]], [[0],[1],[1],[0]])
# print drunken.predict([[0,0],[0,1],[1,0],[1,1]])

# # # DEMO 2 - WINE PREDICTION
from target import *
from training import *

wineFeats = np.array(wineFeats)
targetVals = np.array(targetVals)

trainingFeatures = wineFeats[0:130,:]
trainingTarget = targetVals[0:130,:]

testFeatures = wineFeats[131:,:]
testTarget = targetVals[131:,:]

drunken = Neural((13, 30, 20, 10, 3))
drunken.train(trainingFeatures, trainingTarget, 0.01, 0.01, 100000, True)

print testTarget - drunken.predict(testFeatures) # shows errors in test set
