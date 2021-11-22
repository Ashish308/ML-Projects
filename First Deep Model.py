# -*- coding: utf-8 -*-
import numpy as np
import itertools


trainImages = np.load('fashion_mnist_train_images.npy')
trainLabels = np.load('fashion_mnist_train_labels.npy')
testImages = np.load('fashion_mnist_test_images.npy')
testLabels = np.load('fashion_mnist_test_labels.npy')

def HotCode(y):
    a = np.zeros((y.size,10))
    a[np.arange(y.size),(y)] = 1 
    return a

xTrain = trainImages[:50000,:].T/255
xValidation = trainImages[50000:,:].T/255
xTest = testImages.T/255
yTrain = HotCode(trainLabels[:50000]).T
yValidation = HotCode(trainLabels[50000:]).T
yTest = HotCode(testLabels).T






class NeuralNetwork:
    def __init__(self, xTrain, yTrain, xTest, yTest):
        self.SetHyperParam((0.05, 0.0005, 10, 5, 4, 40))
        self.xTrain, self.yTrain = xTrain, yTrain
        self.xTest, self.yTest = xTest, yTest
    def HeInit(self,x,y):
        return np.random.normal(0,np.sqrt(2/784),(x,y))
    def ReLU(self,x):
        return np.where(x<0,0.0001*x,x)
    
    def DReLU(self,x):
        return np.where(x<0,-0.0001,1)
    
    def Softmax(self,x):
        x -= np.max(x,axis=0)
        return np.exp(x)/np.sum(np.exp(x),axis=0)[np.newaxis]
    
    def SetParam(self):
        if(self.n == 1):
            self.w = [self.HeInit(10,784)]
            self.b = [self.HeInit(10,1)]
        else:
            self.w = [self.HeInit(self.numNeurons,784)] + (self.n-2)*[self.HeInit(self.numNeurons,self.numNeurons)] + [self.HeInit(10,self.numNeurons)]
            self.b = (self.n-1)*[self.HeInit(self.numNeurons,1)] + [self.HeInit(10,1)]
    
    def SetHyperParam(self, hyperParam):
            self.learningRate, self.regularizationFactor, self.batchSize, self.numEpochs, self.n, self.numNeurons = hyperParam
            self.SetParam()
            
    def ForwardPass(self,x):        
        z,h = [None]*self.n,[None]*(self.n-1)
        z[0] = (self.w[0] @ x + self.b[0])
        for i in range(self.n-1):
            h[i] = self.ReLU(z[i])
            z[i+1] = (self.w[i+1] @ h[i] + self.b[i+1])
        yHat = self.Softmax(z[self.n-1])
        return z,h,yHat
    
    def BackPass(self,x,y):
        z,_,yHat = self.ForwardPass(x)
        g = [None]*self.n  
        g[self.n-1] = (yHat - y).T
        for i in range(self.n-1):
            g[self.n-i-2] = (g[self.n-i-1] @ self.w[self.n-i-1]) * self.DReLU(z[self.n-i-2].T)
        return g
    def Update(self,xBatch,yBatch): 
        x,y = xBatch, yBatch 
        _,h,_ = self.ForwardPass(x)
        g = self.BackPass(x,y)
        for i in range(self.n):
          self.b[i] = self.b[i] - self.learningRate * np.mean(g[i],axis=0)[np.newaxis].T
          
        self.w[0] = self.w[0] - self.learningRate * ((g[0].T @ x.T)/x.shape[1] + self.regularizationFactor*self.w[0])
        for i in range(self.n-1):
          self.w[i+1] =  self.w[i+1] - self.learningRate * ((g[i+1].T @ h[i].T)/h[i].shape[1] + self.regularizationFactor*self.w[i+1])

    def SGD(self):
        numExamples = self.xTrain.shape[1]
        numBatch = numExamples//self.batchSize 
        for i in range(self.numEpochs):
            permutation = np.random.permutation(numExamples)
            X,Y = self.xTrain[:,permutation], self.yTrain[:,permutation]
            for j in range(numBatch):
               xBatch = X[:,self.batchSize*j:self.batchSize*(j+1)]
               yBatch = Y[:,self.batchSize*j:self.batchSize*(j+1)]
               self.Update(xBatch,yBatch)

    def Accuracy(self):
        actual = np.argmax(self.yTest,axis=0)
        self.SGD()
        _,_,yHat = self.ForwardPass(self.xTest) 
        result = np.argmax(yHat,axis=0)
        accuracy = np.mean(result==actual)*100
        return accuracy
        
    def ShowAccuracy(self,numSimulations):
        maxAccuracy, minAccuracy = 0,100
        for _ in range(numSimulations): 
            self.SetParam()
            accuracy = self.Accuracy()
            maxAccuracy, minAccuracy = max(accuracy, maxAccuracy), min(accuracy, minAccuracy)
        print("Minimum Accuracy is {:.3f}%".format(minAccuracy))
        print("Maximum Accuracy is {:.3f}%".format(maxAccuracy))
        
    

    
NN = NeuralNetwork(xTrain,yTrain,xTest,yTest)
z,h,_ = NN.ForwardPass(xTrain)
print(NN.Accuracy())

def FindBestHyper():
    BATCH_SIZE = [100,500]
    LEARNING_RATE = [0.075,0.05,0.1]
    NUM_EPOCHS = [2,4]
    REGULARIZATION_FACTOR = [0.05,0.1]
    NUM_HIDDEN_LAYERS = [3,4,5]
    NUM_NEURONS = [25,35,45]
    
    maxAccuracy = 0
    bestHyper = []
    for hyperParam in itertools.product(LEARNING_RATE, REGULARIZATION_FACTOR, BATCH_SIZE, NUM_EPOCHS, NUM_HIDDEN_LAYERS, NUM_NEURONS):
        NN.SetHyperParam(hyperParam)
        if maxAccuracy < NN.Accuracy(): bestHyper = hyperParam
    return bestHyper



