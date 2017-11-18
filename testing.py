from data import *
from train_and_test import *
from ensemble import *

# trainX, trainY = get_synthetic_data("generated_data/well_separated_linear.trn")
# testX, testY = get_synthetic_data("generated_data/well_separated_linear.tst")

# trainX, trainY = get_synthetic_data("generated_data/well_separated_nonlinear.trn")
# testX, testY = get_synthetic_data("generated_data/well_separated_nonlinear.tst")

# trainX, trainY = get_synthetic_data("generated_data/inseparable.trn")
# testX, testY = get_synthetic_data("generated_data/inseparable.tst")

trainX, trainY = get_sat_data("data/sat.trn")
testX, testY = get_sat_data("data/sat.tst")

num_classes = trainY.shape[1]
num_classifiers = 10

## Run each just once
# ens = build_ensemble_1(trainX,trainY,num_classes,epochs=20,num_classifiers=num_classifiers,save=1,save_folder='well_separated_linear')
# ens = build_ensemble_1(trainX,trainY,num_classes,epochs=20,num_classifiers=num_classifiers,save=1,save_folder='well_separated_nonlinear')
# ens = build_ensemble_1(trainX,trainY,num_classes,epochs=20,num_classifiers=num_classifiers,save=1,save_folder='inseparable')
# ens = build_ensemble_1(trainX,trainY,num_classes,epochs=20,num_classifiers=num_classifiers,save=1,save_folder='landsat')

# ens = build_ensemble_from_file(num_classifiers,num_classes,folder_name='well_separated_linear')
# ens = build_ensemble_from_file(num_classifiers,num_classes,folder_name='well_separated_nonlinear')
# ens = build_ensemble_from_file(num_classifiers,num_classes,folder_name='inseparable')
ens = build_ensemble_from_file(num_classifiers,num_classes,folder_name='landsat')

num_classifiers=ens.num_classifiers
k_val = 20 # for some train_and_test

# train_and_test_0(trainX,trainY,testX,testY,0,k_val=10)
# train_and_test_0(trainX,trainY,testX,testY,1,k_val=10)
# train_and_test_1(ens,num_classifiers,num_classes,trainX,trainY,testX,testY)
train_and_test_2_1(ens,num_classifiers,num_classes,trainX,trainY,testX,testY,k_val=k_val)
train_and_test_3(ens,num_classifiers,num_classes,trainX,trainY,testX,testY,k_val=k_val)
# train_and_test_oracle(ens,num_classifiers,num_classes,trainX,trainY,testX,testY)
# train_and_test_single_best(ens,num_classifiers,num_classes,trainX,trainY,testX,testY)
# train_and_test_4(ens,num_classifiers,num_classes,trainX,trainY,testX,testY,k_val=5)

