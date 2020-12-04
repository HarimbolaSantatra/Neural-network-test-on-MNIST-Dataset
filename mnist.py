import numpy as np
import scipy.special
sigmoid = lambda x: scipy.special.expit(x)

class NeuralNetwork():
    def __init__(self,inputNodesNumber,hiddenNodesNumber,outputNodesNumber,learningRate):
        self.inodes = inputNodesNumber
        self.hnodes = hiddenNodesNumber
        self.onodes = outputNodesNumber
        self.lr = learningRate
        self.wih = np.random.normal(0, pow(self.inodes,-0.5), (self.hnodes,self.inodes))
        self.who = np.random.normal(0, pow(self.hnodes,-0.5), (self.onodes,self.hnodes))
    def train(self,inputs,targets):
        # tranposer-na mba ampivadika azy ho colonne; ndmin=2 mba ho afk transposer-na
        valI = np.array(inputs, ndmin=2).T
        valH = sigmoid(np.dot(self.wih ,valI))
        valO = sigmoid(np.dot(self.who , valH))
        targets = np.array(targets, ndmin=2).T
        errorsO = targets - valO
        errorsH = np.dot(np.transpose(self.who) , errorsO)
        self.who += self.lr * np.dot(errorsO * valO * (1 - valO) , np.transpose(valH))
        self.wih += self.lr * np.dot(errorsH * valH * (1 - valH) , np.transpose(valI))
    def query(self,inputs):
        valI = np.array(inputs, ndmin=2).T
        valH = sigmoid(np.dot(self.wih , valI))
        valO = sigmoid(np.dot(self.who , valH))
        reponse = np.argmax(valO.T)
        return reponse

# IMPORT DATA and TAKE 1000 Sample:
trainDataFile = open("mnist_datasets/mnist_train.csv", 'r')
trainFile = trainDataFile.readlines()
trainSample = trainFile[:1000]
trainDataFile.close()

# IMPORT TEST DATA and TAKE 100 Sample:
testDataFile = open("mnist_datasets/mnist_test.csv", 'r')
testFile = testDataFile.readlines()
testSample = testFile[:100]
testDataFile.close()

# CREATE AN INSTANCE OF THE NEURAL NETWORK CLASS:
mnistNN = NeuralNetwork(784,100,10,0.2)

# TRAIN :
for line in trainSample:
    arrayedSample = line.split(",")
    targets = np.zeros(10)
    targets[int(arrayedSample[0])] = 1
    train_inputs = np.asfarray(arrayedSample[1:])
    train_inputs = (((train_inputs/255)*0.99)+0.01)
    mnistNN.train(train_inputs,targets)

# QUERY :
score = 0
for line in testSample:
    line = line.split(",")
    line = np.asfarray(line)
    testLabel = line[0]
    test_inputs = line[1:]

    # COMPUTE SCORE:
    if (mnistNN.query(test_inputs) == testLabel):
        score+=1

print("TEST SCORE: " + str(score/len(testSample)))