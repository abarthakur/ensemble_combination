from data import *
from train_and_test import *
from ensemble import *


trainX, trainY = get_sat_data("data/sat.trn")
testX, testY = get_sat_data("data/sat.tst")
num_classes = trainY.shape[1]

ens = build_ensemble_kuncheva_sat(trainX,trainY,num_classes)
num_classifiers=ens.num_classifiers

train_and_test_1(ens,num_classifiers,num_classes,trainX,trainY,testX,testY)
train_and_test_2_1(ens,num_classifiers,num_classes,trainX,trainY,testX,testY,k_val=10)
train_and_test_0(trainX,trainY,testX,testY,0,k_val=10)
train_and_test_0(trainX,trainY,testX,testY,1,k_val=10)
train_and_test_3(ens,num_classifiers,num_classes,trainX,trainY,testX,testY,k_val=10)
# train_and_test_4(ens,num_classifiers,num_classes,trainX,trainY,testX,testY,k_val=5)
train_and_test_oracle(ens,num_classifiers,num_classes,trainX,trainY,testX,testY)
train_and_test_single_best(ens,num_classifiers,num_classes,trainX,trainY,testX,testY)




