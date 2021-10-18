

import numpy as np


a=np.longdouble(np.load('age_regression_Xtr.npy'))
b=np.longdouble(np.load('age_regression_ytr.npy'))
c=np.longdouble(np.load('age_regression_Xte.npy'))
d=np.longdouble(np.load('age_regression_yte.npy'))


xTrain = a[:4000,:,:].reshape(4000,2304).T
yTrain = b[:4000][np.newaxis].T
xValidation = a[4000:,:].reshape(1000,2304).T
yValidation = b[4000:][np.newaxis].T


BATCH_SIZE = [2,5,10,100]
LEARNING_RATE = [0.001,0.01,0.1,1]
EPOCHS = [1,2,3,4]
REGULARIZATION_FACTOR = [0.01,0.1,1,10]

def MSE(X,y,w,b):
    return np.mean(np.square(X.T@w+b-y))/2




def Update(X,y,w,b,learningRate,regularizationFactor):
    W =  w-learningRate * np.mean(X@(X.T@w+b - y) + regularizationFactor*w,axis=1)[np.newaxis].T
    B = b - learningRate * np.mean(X.T@w+b - y)
    return W,B

    
def SGD(numEpochs,batchSize,learningRate,regularizationFactor):
    numBatch = 4000//batchSize 
    w,b = np.ones(2304)[np.newaxis].T,1
    for i in range(numEpochs):
	permutation = np.random.permutation(50000)
        X = xTrain[:,permutation]
        Y = yTrain[permutation,:]           
        for j in range(numBatch):
           xBatch = xTrain[:,batchSize*j:batchSize*(j+1)]
           yBatch = yTrain[batchSize*j:batchSize*(j+1)]
           w,b = Update(xBatch,yBatch,w,b,learningRate,regularizationFactor)
    return MSE(xValidation,yValidation,w,b),w,b





minMSE = float('inf')
wFinal = np.ones(2304)[np.newaxis].T,1
bFinal = 1
numEpochsBest,batchSizeBest,learningRateBest,regularizationFactorBest = 1,1,1,1

for batchSize in BATCH_SIZE:
    for learningRate in LEARNING_RATE:
        for numEpochs in EPOCHS:
            for regularizationFactor in REGULARIZATION_FACTOR:
                mse,w,b = SGD(numEpochs,batchSize,learningRate,regularizationFactor)
                if(mse<minMSE):
                    minMSE = mse
                    wBest = w
                    bBest = b
                    numEpochsBest,batchSizeBest,learningRateBest,regularizationFactorBest = numEpochs,batchSize,learningRate,regularizationFactor

print("The MSE of the test set is: {}".format(minMSE))
print('')
print("Best Hyperparameters:")
print("Number of Epochs: {}".format(numEpochsBest))
print("Batch Size: {}".format(batchSizeBest))
print("Learning Rate: {}".format(learningRateBest))
print("Regularization Factor: {}".format(regularizationFactor))
print('')
print("Best Model: y = {}x + {}".format(wBest,bBest))
