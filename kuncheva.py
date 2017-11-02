from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import numpy as np
from dt_2 import *

#not sure this is necessary yet, but oh well
class QDCClassifier :

	#wrap trained model in this class
	def __init__(self,model,fset,num_classes):
		self.model=model
		self.fset=fset
		self.num_classes=num_classes

	def predict(self,sample):
		return self.model.predict_proba(np.array([sample[self.fset]])).reshape((1,self.num_classes))


#build ensemble
def build_ensemble_kuncheva_sat(trainX,trainY,num_classes):
	###############Build ensemble of 6 QDCs trained on two features each#####################
	ens = Ensemble(num_classes)
	trainY=np.argmax(trainY,axis=1)

	total_samples = trainX.shape[0]
	features=[[16,17],[16,18],[16,19],[17,18],[17,19],[18,19]]

	num_classifiers=len(features)

	for fset in features:
		clf = QuadraticDiscriminantAnalysis()
		tX=trainX[:,np.array(fset)]
		#print(tX[1:5,:])
		#print(trainX[1:5,:])
		clf.fit(tX, trainY)
		ens.add_classifier(QDCClassifier(clf,fset,num_classes))
	return ens

#get training data
trainX, trainY = get_data("data/sat.trn")
testX, testY = get_data("data/sat.tst")

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
