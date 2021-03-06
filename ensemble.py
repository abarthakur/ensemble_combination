import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json # need python-h5py (or python3-h5py) installed for this
import math

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


class Ensemble :

	def __init__(self,num_classes):
		self.num_classifiers=0
		self.num_classes=num_classes
		self.classifier_list=[]

	def add_classifier(self,classifier):
		self.classifier_list.append(classifier)
		self.num_classifiers=self.num_classifiers+1


	def get_profile(self,sample):
		#Decision profile
		DP = np.zeros([self.num_classifiers, self.num_classes])
		for i in range(0,self.num_classifiers):
			#base classifier type
			classifier=self.classifier_list[i]
			#numpy array size 1 x num_classes
			prediction=classifier.predict(sample)
			DP[i,:]=prediction
		return DP



class KerasClassifier :

	#wrap trained model in this class
	def __init__(self,model,num_classes):
		self.model=model
		self.num_classes=num_classes

	def predict(self,sample):
		return self.model.predict(np.array([sample])).reshape((1,self.num_classes))



#build ensemble
def build_ensemble_1(trainX,trainY,num_classes,epochs,num_classifiers=10,save=0,save_folder=''):
	###############Build ensemble of MLPs with bagging (sort of)#####################
	ens = Ensemble(num_classes)
	half=int(num_classifiers/2)

	#employ bagging
	total_samples = trainX.shape[0]
	max_samples = (total_samples/2)
	max_samples=int(math.ceil(max_samples))

	#NN with single hidden layer,trained on all features x 5
	for i in range(0,half):
		bag = np.random.randint(low=0, high=total_samples-1, size=max_samples)
		tX_part= trainX[bag]
		tY_part= trainY[bag]
		model1 = Sequential()
		model1.add(Dense(8, input_dim=trainX.shape[1], activation="sigmoid"))
		model1.add(Dense(num_classes, activation="softmax"))
		model1.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
		model1.fit(tX_part,tY_part, epochs=epochs, batch_size=10) # epochs=150

		if(save):
			save_model_to_file(model1,i,save_folder)
			print("Saved model to disk")

		ens.add_classifier(KerasClassifier(model1,num_classes))

	#NN with 2 hidden layers x 5
	for i in range(half,num_classifiers):
		bag = np.random.randint(low=0, high=total_samples-1, size=max_samples)
		tX_part= trainX[bag]
		tY_part= trainY[bag]
		model2 = Sequential()
		model2.add(Dense(8, input_dim=trainX.shape[1], activation="sigmoid"))
		model2.add(Dense(8, activation="sigmoid"))
		model2.add(Dense(num_classes, activation="softmax"))
		model2.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
		model2.fit(tX_part, tY_part, epochs=epochs, batch_size=10) # epochs=150

		if(save):
			save_model_to_file(model2,i,save_folder)
			print("Saved model to disk")

		ens.add_classifier(KerasClassifier(model2,num_classes))
	#################################################################3
	return ens


#build ensemble - load models from file
def build_ensemble_from_file(num_classifiers,num_classes,folder_name):
	ens = Ensemble(num_classes)
	# num_classifiers=10
	half=int(num_classifiers/2)

	for i in range(num_classifiers):
		filename = 'saved_models/'+folder_name+'/'+str(i)
		json_file = open(filename+'.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		loaded_model = model_from_json(loaded_model_json) # need python-h5py (or python3-h5py) installed for this
		# load weights into new model
		loaded_model.load_weights(filename+'.h5')

		ens.add_classifier(KerasClassifier(loaded_model,num_classes))
	print("Loaded models from disk")

	return ens


def save_model_to_file(model,id,folder_name):
	filename = 'saved_models/'+folder_name+'/'+str(id)
	model_json = model.to_json()
	with open(filename+'.json', "w") as json_file:
		json_file.write(model_json)
	model.save_weights(filename+'.h5')


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