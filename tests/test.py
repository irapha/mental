import sys
sys.path.append("..")
from mental.mental import Neural
import numpy as np

# # # DEMO 1 - XOR
# xor = Neural((2, 3, 1))
# xor.train([[0,0],[0,1],[1,0],[1,1]], [[0],[1],[1],[0]])
# print xor.predict([[0,0],[0,1],[1,0],[1,1]])



# # # DEMO 2 - WINE PREDICTION
# from target import *
# from training import *
#
# wineFeats = np.array(wineFeats)
# targetVals = np.array(targetVals)

# trainingFeatures = wineFeats[0:130,:]
# trainingTarget = targetVals[0:130,:]
#
# testFeatures = wineFeats[131:,:]
# testTarget = targetVals[131:,:]

# drunken = Neural((13, 30, 20, 10, 3))
# drunken.train(trainingFeatures, trainingTarget, 0.01, 0.01, 100000, True)
# print testTarget - drunken.predict(testFeatures) # shows errors in test set



# # # DEMO3 - plotting Jtrain and Jcv for wine prediction
# import matplotlib.pyplot as plt
# from target import *
# from training import *
#
# wineFeats = np.array(wineFeats)
# targetVals = np.array(targetVals)
#
# trainFeats = wineFeats[0:88]
# trainTargs = targetVals[0:88]
#
# cvFeats = wineFeats[89:-1]
# cvTargs = targetVals[89:-1]
#
# trainCosts = []
# cvCosts = []
#
# # training many networks. Should take ~4.5 min
# for i in range(2, trainFeats.shape[0], 6):
#     net = Neural((13, 30, 20, 10, 3))
#     net.train(trainFeats[:i], trainTargs[:i], 0.01, 0.01, 50000)
#     trainCosts.append(net.cost(trainFeats[:i], trainTargs[:i], 0.01))
#     cvCosts.append(net.cost(cvFeats[:i], cvTargs[:i], 0.01))
#
# plt.plot(trainCosts)
# plt.plot(cvCosts)
# plt.show()
