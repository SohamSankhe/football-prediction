import numpy as np

def divideListForTrnTest(lst, trainingLimit):

    trainingLimit = lst.__len__() - int(lst.__len__() * trainingLimit)
    print('trainingLimit: ', trainingLimit)

    trainingList = lst[:trainingLimit]
    testingList = lst[trainingLimit:]

    print('np.shape(trainingList): ', np.shape(trainingList))
    print('np.shape(testingList): ', np.shape(testingList))
    # print(trainingList)
    # print(testingList)

    return trainingList, testingList


def getCommaSepForm(lst):
    'Get elements in comma separated form'

    csString = "\'"
    csString += "\',\'".join(map(str, lst))
    csString += "\'"

    return csString


def getTargetVariables(oldY):
    'Convert Y from 0,1,-1 to -1,1'

    Y = []
    for i in range(0, oldY.__len__()):
        if (oldY[i] == 0) or (oldY[i] == -1):
            Y.append(0)
        else:
            Y.append(1)
    return Y