import numpy as np

a = np.load('fashion_mnist_train_images.npy')
b = np.load('fashion_mnist_train_labels.npy')
c = np.load('fashion_mnist_test_images.npy')
d = np.load('fashion_mnist_test_labels.npy')



def HotCode(y):
    a = np.zeros((y.size,10))
    a[np.arange(y.size),(y)] = 1 
    return a

xTrain = a[:50000,:].T/255
xValidation = a[50000:,:].T/255
xTest = c.T/255
yTrain = HotCode(b[:50000])
yValidation = HotCode(b[50000:])
yTest = HotCode(d)





BATCH_SIZE = [2,5,10,100]
LEARNING_RATE = [0.001,0.01,0.1,1]
EPOCHS = [1,2,3,4]
REGULARIZATION_FACTOR = [0.01,0.1,1,10]



    
def Model(X,W,B):
    Z = X.T@W + B
    Y_hat = np.exp(Z)/np.sum(np.exp(Z),axis=1)[np.newaxis].T
    return Y_hat
#Cross Entropy
def CE(Y_hat,Y):
    return -np.mean(Y*np.log(Y_hat))


def Update(X,Y,W,B,learningRate,regularizationFactor):
    Y_hat = Model(X,W,B)
    W_new =  W-learningRate * (X@(Y_hat - Y)/X.shape[1] + regularizationFactor*W)
    B_new = B - learningRate * np.mean(Y_hat - Y,axis=0)
    return W_new,B_new


def SGD(numEpochs,batchSize,learningRate,regularizationFactor):
    numBatch = 50000//batchSize 
    W,B = np.ones((784,10)),np.ones(10)
    for i in range(numEpochs):
        permutation = np.random.permutation(50000)
        X = xTrain[:,permutation]
        Y = yTrain[permutation,:]        
        for j in range(numBatch):
           xBatch = X[:,batchSize*j:batchSize*(j+1)]
           yBatch = Y[batchSize*j:batchSize*(j+1),:]    
           W,B = Update(xBatch,yBatch,W,B,learningRate,regularizationFactor) 
    return W,B


# 2,100,0.1,0.1 are the best hyperparameters
#Code for determining best hyperparameters
'''
minCE = np.Inf

for batchSize in BATCH_SIZE:
    for learningRate in LEARNING_RATE:
        for numEpochs in EPOCHS:
            for regularizationFactor in REGULARIZATION_FACTOR:
                CrossEntropy = SGD(numEpochs,batchSize,learningRate,regularizationFactor)
                if(CrossEntropy<minCE):
                    minCE = CrossEntropy

                    numEpochsBest,batchSizeBest,learningRateBest,regularizationFactorBest = numEpochs,batchSize,learningRate,regularizationFactor

print(numEpochsBest,batchSizeBest,learningRateBest,regularizationFactorBest)
'''


w,b = SGD(2,100,0.1,0.1)
y_hat = Model(xTest,w,b)
result = np.argmax(y_hat,axis=1)[np.newaxis].T
d=d.reshape(-1,1)

print('Cross Entropy: {}'.format(CE(y_hat,yTest)))
print('Accuracy: {:.3f}%'.format(np.mean(result==d)*100))
