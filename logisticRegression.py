import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import utils

# Call logisticRef method

def logit(z):
    sigma = 1 / (1 + np.exp(-z))
    return sigma


def likelihood(X, theta, Y):
    'loss - Minimize using gd'
    yRows = np.shape(Y)[0]
    z = X.dot(theta)
    l = logit(z)
    loss = ((-Y * np.log(l) - (1 - Y) * np.log(1 - l))) / yRows
    return loss


def getGradient(X, Y, theta):
    z = X.dot(theta)
    l = logit(z)
    grad = X.transpose().dot(l - Y) / np.shape(Y)[0]
    return grad


def getPrediction(X, theta):
    z = X.dot(theta)
    pred = logit(z)
    roundedPred = []
    for p in pred:
        if p <= 0.5:
            roundedPred.append(0)
        else:
            roundedPred.append(1)
    return roundedPred


def checkAccuracy(xTest, yTest, theta):
    correctCtr = 0  # number of correct predictions
    prediction = getPrediction(xTest, theta)

    for i in range(0, xTest.__len__()):
        if prediction[i] == yTest[i]:
            correctCtr += 1

    correctCtr = correctCtr / prediction.__len__()

    return correctCtr

'''
def checkAccuracySkLearn(xTest, yTest):
    correctCtr = 0  # number of correct predictions
    prediction = getPrediction(xTest, theta)

    for i in range(0, xTest.__len__()):
        if prediction[i] == yTest[i]:
            correctCtr += 1

    correctCtr = correctCtr / prediction.__len__()

    return correctCtr
'''

def calculateThetaGradDesc(X, Y):
    learningRate = 0.1
    theta = np.zeros(np.shape(X)[1])  # d*1

    for i in range(0, 100):

        #print(np.shape(X))

        gradient = getGradient(X, Y, theta)
        theta = theta - (learningRate * gradient)
        loss = likelihood(X, theta, Y)

        l1Error = sum(pow(loss, 2)) / loss.__len__()

    return theta


def logisticReg(xTraining,yTraining,xTest, yTest):

    theta = calculateThetaGradDesc(xTraining, yTraining)
    print('Theta: ', theta)
    print('Training accuracy: ', checkAccuracy(xTraining, yTraining, theta))
    print('Test accuracy: ', checkAccuracy(xTest, yTest, theta))

    '''
    # sklearn
    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class = 'multinomial').fit(xTraining, yTraining)
    print('clf.predict(xTest): \n', clf.predict(xTest))
    print('clf.predict_proba(X):\n', clf.predict_proba(xTest))
    print(clf.score(xTest, yTest))
    '''

    return theta


def getTargetVarMultiClass(oldY, res):

    Y = []

    for i in range(0, oldY.__len__()):
        if oldY[i] == res:
            Y.append(1)
        else:
            Y.append(0)
    return Y


def getPredictionMultiClass(xTest, wList):

    predProbabilites = []

    for w in wList:
        z = xTest.dot(w)
        pred = logit(z)
        predProbabilites.append(pred)

    # print(predProbabilites[0])
    # print(predProbabilites[1])
    # print(predProbabilites[2])

    prediction = []

    for i in range(pred.__len__()):

        if (predProbabilites[0][i] > predProbabilites[1][i]) and (predProbabilites[0][i] > predProbabilites[2][i]):
            prediction.append(1)  # win
        elif (predProbabilites[1][i] > predProbabilites[0][i]) and (predProbabilites[1][i] > predProbabilites[2][i]):
            prediction.append(0)  # draw
        else:
            prediction.append(-1)  # lose

    return prediction


def checkAccuracyMultiClass(xTest, yTest, wList):

    prediction = getPredictionMultiClass(xTest, wList)

    #print('prediction:\n', prediction)
    #print(yTest)

    correctCtr = 0
    for i in range(0, xTest.__len__()):
        if prediction[i] == yTest[i]:
            correctCtr += 1

    correctCtr = correctCtr / prediction.__len__()

    return correctCtr



def classifyLogistic(xTraining, yTraining, xTest, yTest):

    print('Logistic regression:')
    wList = []
    resultTuple = (1, 0, -1) # 1- win, 0 - draw, -1 - loss

    for r in resultTuple:

        yTrn = np.mat(getTargetVarMultiClass(yTraining, r))
        yTst = np.mat(getTargetVarMultiClass(yTest, r))

        # getting in required format
        xTrn = np.squeeze(np.asarray(xTraining))
        yTrn = np.squeeze(np.asarray(yTrn))
        xTst = np.squeeze(np.asarray(xTest))
        yTst = np.squeeze(np.asarray(yTst))

        w = logisticReg(xTrn, yTrn.transpose(), xTst, yTst.transpose())

        wList.append(w)

    # check accuracy on wList
    xTest = np.squeeze(np.asarray(xTest))
    yTest = np.squeeze(np.asarray(yTest))
    print('Accuracy of multi class logistic: ', checkAccuracyMultiClass(xTest, yTest.transpose(), wList) * 100, '%')

    return wList


def main():
    dataSet = sio.loadmat('.\\dataset3.mat', squeeze_me=True)
    xTraining = dataSet['X_trn']
    yTraining = dataSet['Y_trn']
    xTest = dataSet['X_tst']
    yTest = dataSet['Y_tst']

    print('Logistic regression: \n')

    print('For dataset3.mat')
    theta = calculateThetaGradDesc(xTraining, yTraining)
    print('Theta: ', theta)

    print('np.shape(xTraining): ',np.shape(xTraining))
    print('np.shape(yTraining): ', np.shape(yTraining))
    print('np.shape(theta): ',np.shape(theta))
    print(type(xTraining))
    print(type(yTraining))

    print('Training accuracy: ', checkAccuracy(xTraining, yTraining, theta))
    print('Test accuracy: ', checkAccuracy(xTest, yTest, theta))

    print('\nFor dataset4.mat')

    dataSet = sio.loadmat('.\\dataset4.mat', squeeze_me=True)
    xTraining = dataSet['X_trn']
    yTraining = dataSet['Y_trn']
    xTest = dataSet['X_tst']
    yTest = dataSet['Y_tst']

    theta = calculateThetaGradDesc(xTraining, yTraining)
    print('Theta: ', theta)

    print('Training accuracy: ', checkAccuracy(xTraining, yTraining, theta))
    print('Test accuracy: ', checkAccuracy(xTest, yTest, theta))

    print('\nNote: Accuracy is calculated as (correct predictions / total predictions)')
    # unable to plot hyperplane

    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(xTraining, yTraining)
    print('clf.predict(xTest): \n', clf.predict(xTest))
    print('clf.predict_proba(X):\n', clf.predict_proba(xTest))
    print(clf.score(xTest, yTest))

    return


#main()

