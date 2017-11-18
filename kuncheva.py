import numpy as np
from train_and_test import *
from data import *
from ensemble import *

#get training data
trainX, trainY = get_sat_data("data/sat.trn")
testX, testY = get_sat_data("data/sat.tst")

num_classes = trainY.shape[1]

#combine both
allX = np.concatenate((trainX,testX),axis=0)
allY=np.concatenate((trainY,testY),axis=0)
#shuffle them
permutation = np.random.permutation(allX.shape[0])
allX=allX[permutation,:]
allY=allY[permutation,:]

n=2000

trainX=allX[:n,:]
testX=allX[n:,:]
trainY=allY[:n,:]
testY=allY[n:,:]


#print("here")

ens = build_ensemble_kuncheva_sat(trainX,trainY,num_classes)
num_classifiers=ens.num_classifiers
train_and_test_1(ens,num_classifiers,num_classes,trainX,trainY,testX,testY)
train_and_test_0(trainX,trainY,testX,testY,0,k_val=10)
train_and_test_0(trainX,trainY,testX,testY,1,k_val=10)
