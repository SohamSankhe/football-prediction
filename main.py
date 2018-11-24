import dbConnector
import sqlQueries
import pandas as pd
import numpy as np
import utils
import teamRatingPredictionData
import linearRegression
import logisticRegression
from sklearn.linear_model import LogisticRegression
import sys
import matchOutcomeClassificationData

TRAINING_LIMIT = 0.05 # percent of data
CLASSIFICATION_TRAINING_LIMIT = 0.05 # percent of data

# get all match ids
con = dbConnector.getConnection()
match_ids = pd.read_sql(sqlQueries.MATCHES, con)

# divide matches into training and testing
matchList = list(match_ids.values.flatten())
trainingMatches, testingMatches = utils.divideListForTrnTest(matchList, TRAINING_LIMIT)

# prepare data for linear regression
print('Loading data for linear regression')
xTrainingReg, yTrainingReg = teamRatingPredictionData.getRegressionDataTraining(trainingMatches)
xTestReg, yTestReg, teamRatingTestMat = teamRatingPredictionData.getRegressionDataTest(testingMatches)

# regression for team ratings
lambdaList = [0.01, 0.05, 0.1, 0.25, 0.5, 1]
kList = [2, 10]

# train
xTrainingReg = np.squeeze(np.asarray(xTrainingReg))
yTrainingReg = np.squeeze(np.asarray(yTrainingReg))
xTestReg = np.squeeze(np.asarray(xTestReg))
yTestReg = np.squeeze(np.asarray(yTestReg))

print('Getting theta for predicting team ratings')
#thetaRidge = linearRegression.ridgeRegression(xTrainingReg, yTrainingReg.transpose(), xTestReg, yTestReg.transpose(), lambdaList, kList)
thetaLinear = linearRegression.linearReg(xTrainingReg, yTrainingReg.transpose(), xTestReg, yTestReg.transpose())

#sseTestRidge = linearRegression.calculateError(xTestReg, thetaRidge, yTestReg.transpose())
sseTestLinear = linearRegression.calculateError(xTestReg, thetaLinear, yTestReg.transpose())

#print('sseTestRidge: ', sseTestRidge)
print('sseTestLinear: ', sseTestLinear)

# pick theta which gives min error on test
theta = thetaLinear
#if sseTestRidge < sseTestLinear:
#    theta = thetaRidge

print('\ntheta: ', theta)


# Use theta to predict team ratings for matches test
teamRatings = xTestReg.dot(theta)

#for i in range(teamRatings.__len__()):
#    print(teamRatings[i], '-', yTestReg[i])

matchIdCol = np.squeeze(np.asarray(teamRatingTestMat[:,0]))
teamIdCol = np.squeeze(np.asarray(teamRatingTestMat[:,1]))

# Use for team ratings in test
teamRatingsForTest = pd.DataFrame({'match_id':matchIdCol,'team_id':teamIdCol, 'actual_rating':yTestReg, \
                        'predicted_rating': teamRatings})

#print('teamRatingsForTest: \n ', teamRatingsForTest)

# how to map predicted team ratings to matches & team ids
# order by matchid and team id in query enough?
# make dataframe with rows -> matchid, teamid, team_rating (use for mapping during class)

# classification

# get matchdata for training matches
reducedMatchData = matchOutcomeClassificationData.getMatchDataTraining(trainingMatches)

# divide match training data further into training and testing
xTrainingClass, yTrainingClass, xTestClass, yTestClass = \
    matchOutcomeClassificationData.divideClassificationData(reducedMatchData, CLASSIFICATION_TRAINING_LIMIT)

# train classifier
wListLogistic = logisticRegression.classifyLogistic(xTrainingClass, yTrainingClass, xTestClass, yTestClass)

# get matchdata for testing matches
_, matchTestingData = matchOutcomeClassificationData.getMatchDataTesting(testingMatches)

# matchTestingData does not have team ratings - has 0 value instead
# use predicted team rating in 'teamRatingsForTest' to fill missing values

matchTestingData = matchOutcomeClassificationData.mergeTeamRatingsAndMatchStats(matchTestingData, teamRatingsForTest)

rowsToIgnore = ['match_id','home_team_id', 'away_team_id','full_time_score']
reducedMatchTestingData = matchTestingData.drop(rowsToIgnore, axis=1) # feature matrix for outcome pred

rows, columns = np.shape(reducedMatchTestingData)
reducedMatchTestingData = np.mat(reducedMatchTestingData)
xMatchClassTest = reducedMatchTestingData[:, 0:columns-1]
yMatchClassTest = reducedMatchTestingData[:, columns-1]

xMatchClassTest = np.squeeze(np.asarray(xMatchClassTest))
yMatchClassTest = np.squeeze(np.asarray(yMatchClassTest))

print('Accuracy with logistic: ', \
      logisticRegression.checkAccuracyMultiClass(xMatchClassTest, yMatchClassTest.transpose(), wListLogistic) * 100, '%')

'''
import svmSmo
wListSmo = svmSmo.customSvmMultiClass(xTrainingClass, yTrainingClass, xTestClass, yTestClass)
xTestSvm = reducedMatchTestingData[:, 0:columns-1]
yTestSvm = reducedMatchTestingData[:, columns-1]
print('Accuracy with SVM: ', \
      svmSmo.checkAccuracyMultiClass(xTestSvm, yTestSvm, xTrainingClass, yTrainingClass, wListSmo) * 100, '%')
'''

