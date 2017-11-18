import numpy as np

import math

from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K

from sklearn.neighbors import NearestNeighbors


#build decision templates
def build_decision_templates(ens,num_classifiers,num_classes,trainX,trainY):
	DT = np.zeros([num_classes, num_classifiers, num_classes])
	counts = np.zeros(num_classes)

	for i in range(trainX.shape[0]):
		# class to which trainX[i] belongs
	    c=np.argmax(trainY[i])
	    DT[c] = DT[c] + ens.get_profile(trainX[i])
	    counts[c] += 1

	for c in range(DT.shape[0]):
		if counts[c] !=0:
			DT[c] = np.divide(DT[c], counts[c])
		else :
			# print("Oops empty!")
			DT[c]=DT[c]+np.inf

	return DT

####Prediction rule 0 : simple kNN classifier with majority voting
def train_and_test_0(trainX,trainY,testX,testY,voting_choice=0,k_val=5):
	#build nearest neighbours data structure for efficient search
	nbrs = NearestNeighbors(n_neighbors=k_val, algorithm='ball_tree').fit(trainX)

	#predict
	tot_test=testX.shape[0]
	# tot_test=10
	correct=0

	for i in range(0,tot_test):
		sample=testX[i]

		[dist_nbrs,indices]= nbrs.kneighbors(sample.reshape(1,-1))
		# print(indices)
		X_nbrs = trainX[indices[0]]
		Y_nbrs= trainY[indices[0]]

		if voting_choice==0:
			total_votes = np.sum(Y_nbrs,axis=0)
		else:
			# print(np.reshape(dist_nbrs[0],(-1,1)))
			# print(Y_nbrs)
			temp=np.divide(Y_nbrs,np.reshape(dist_nbrs[0],(-1,1)))
			# print(temp)
			total_votes = np.sum(temp,axis=0)
		# print(total_votes)

		label = np.argmax(total_votes)
		true_label=np.argmax(testY[i])

		# print("Prediction :"+str(label))
		# print("Actual :"+str(true_label))
		if label==true_label:
			correct+=1

	accuracy=float(correct)/float(tot_test)
	print("Prediction rule 0")
	print("Total test accuracy is :")
	print(accuracy)
	return accuracy

####Prediction rule 1 : Decision Templates (Kuncheva 2001)
def train_and_test_1(ens,num_classifiers,num_classes,trainX,trainY,testX,testY):

	DT=build_decision_templates(ens,num_classifiers,num_classes,trainX,trainY)
	#predict
	tot_test=testX.shape[0]
	correct=0

	for i in range(0,tot_test):
		sample=testX[i]
		DP= ens.get_profile(sample)
		distances = np.sum(((DT - DP)**2),axis=(1,2))
		label = np.argmin(distances,axis=(0))
		true_label=np.argmax(testY[i])
		# print("Prediction :"+str(label))
		# print("Actual :"+str(true_label))
		if label==true_label:
			correct+=1

	accuracy=float(correct)/float(tot_test)
	print("Prediction rule 1")
	print("Total test accuracy is :")
	print(accuracy)
	return accuracy

####Prediction rule 2.1 : create DTs from the kNN in feature space
def train_and_test_2_1(ens,num_classifiers,num_classes,trainX,trainY,testX,testY,k_val=5):
	#build nearest neighbours data structure for efficient search
	nbrs = NearestNeighbors(n_neighbors=k_val, algorithm='ball_tree').fit(trainX)
	#predict
	tot_test=testX.shape[0]
	# tot_test=10
	correct=0
	for i in range(0,tot_test):
		sample=testX[i]

		[dist_nbrs,indices]= nbrs.kneighbors(sample.reshape(1,-1))
		# print(indices)
		X_nbrs = trainX[indices[0]]
		Y_nbrs= trainY[indices[0]]

		#build decision templates
		DT_nbrs=build_decision_templates(ens,num_classifiers,num_classes,X_nbrs,Y_nbrs)
		DP= ens.get_profile(sample)
		distances = np.sum(((DT_nbrs - DP)**2),axis=(1,2))

		label = np.argmin(distances,axis=(0))
		true_label=np.argmax(testY[i])
		# print("Prediction :"+str(label))
		# print("Actual :"+str(true_label))
		if label==true_label:
			correct+=1

	accuracy=float(correct)/float(tot_test)
	print("Prediction rule 2.1")
	print("Total test accuracy is :")
	print(accuracy)
	return accuracy

####Prediction rule 3 : find k-NN in DP space. find which classifiers did well for the corresponding points,
####					and then weigh them accordingly ending in a weighted majority voting
def train_and_test_3(ens,num_classifiers,num_classes,trainX,trainY,testX,testY,k_val=5):
	#build the decision profile list
	DP_X_flat = np.zeros([trainX.shape[0],num_classifiers*num_classes])
	for i in range(0,trainX.shape[0]):
		DP_X_flat[i]= np.reshape(ens.get_profile(trainX[i]),num_classifiers*num_classes)

	#build nearest neighbours data structure for efficient search
	nbrs = NearestNeighbors(n_neighbors=k_val, algorithm='ball_tree').fit(DP_X_flat)
	#predict
	tot_test=testX.shape[0]
	# tot_test=10
	correct=0
	for i in range(0,tot_test):
		sample=testX[i]
		DP=ens.get_profile(sample)
		DP_flat= np.reshape(DP,(1,num_classifiers*num_classes))

		[dist_dp_nbrs,indices]= nbrs.kneighbors(DP_flat)

		X_nbrs = trainX[indices[0]]
		Y_nbrs= trainY[indices[0]]
		Y_nbrs_labels= np.argmax(Y_nbrs,axis=1)

		classifier_scores = np.zeros(num_classifiers)

		for j in range(0,X_nbrs.shape[0]):
			xnbr= X_nbrs[j]
			dp_xnbr=ens.get_profile(xnbr)
			# print(dp_xnbr)
			predictions = np.argmax(dp_xnbr,axis=1)
			# print(predictions)
			# print(Y_nbrs_labels[i])
			truth = np.zeros(num_classifiers)+Y_nbrs_labels[j]
			# print(predictions==truth)
			classifier_scores=classifier_scores+(predictions==truth)
			# print(classifier_scores)
			# break
		
		# print(DP)
		predictions = np.argmax(DP,axis=1)
		predictions_one_hot= np.zeros([num_classifiers,num_classes]) 
		for j in range(0,num_classifiers):
			predictions_one_hot[j][predictions[j]]=1
		# print(predictions_one_hot)
		# print(classifier_scores)
		weighted_votes= np.multiply(predictions_one_hot,np.reshape(classifier_scores,(-1,1)))
		# print(weighted_votes)
		total_votes=np.sum(weighted_votes,axis=0)
		# print(total_votes)
		label = np.argmax(total_votes,axis=(0))
		true_label=np.argmax(testY[i])

		# print("Prediction :"+str(label))
		# print("Actual :"+str(true_label))
		if label==true_label:
			correct+=1

	accuracy=float(correct)/float(tot_test)
	print("Prediction rule 3")
	print("Total test accuracy is :")
	print(accuracy)
	return accuracy

####Prediction rule 4 : Given a sample, for each classifier, find the kNN in the projection space of the NN. See how well the classifier
####					performs on these kNN, and weight the classifier accordingly
def train_and_test_4(ens,num_classifiers,num_classes,trainX,trainY,testX,testY,k_val=5):
	projections = []
	nbrs = []

	for keras_classifier in ens.classifier_list:
		model = keras_classifier.model
		get_proj = K.function([model.layers[0].input],
	                                  [model.layers[-2].output])
		proj = get_proj([trainX])[0]
		projections.append(proj)
		nbrs.append( NearestNeighbors(n_neighbors=k_val, algorithm='ball_tree').fit(proj) )

	# #predict
	tot_test=testX.shape[0]
	# # tot_test = 30
	correct=0

	for i in range(0,tot_test):
		sample=testX[i]
		sample=np.reshape(sample,(1,sample.shape[0]))
		classifier_scores = np.zeros(num_classifiers)
		#for each classifier, find k-NN in proj space
		for c in range(num_classifiers):
			model= ens.classifier_list[c].model
			get_proj = K.function([model.layers[0].input],
	                                  [model.layers[-2].output])
			# print("shape")
			proj = get_proj([sample])[0]
			[dist_dp_nbrs,indices]= nbrs[c].kneighbors(proj)

			X_nbrs = trainX[indices[0]]
			Y_nbrs= trainY[indices[0]]

			#check which nbrs the classifier did well on
			for j in range(0,X_nbrs.shape[0]): #to get prediction of classifier c for nbrs
				xnbr= X_nbrs[j]
				proj_xnbr_c=ens.get_profile(xnbr)[c] #num_classes size
				prediction = np.argmax(proj_xnbr_c)
				truth = np.argmax(Y_nbrs[j]) #because Y labels are one-hot
				classifier_scores[c] += int(prediction==truth)

		# print(sample.shape)
		sample=testX[i]
		DP=ens.get_profile(sample)
		predictions = np.argmax(DP,axis=1) #by each classifier
		predictions_one_hot= np.zeros([num_classifiers,num_classes])
		for j in range(0,num_classifiers):
			predictions_one_hot[j][predictions[j]]=1
		
		weighted_votes = np.multiply(predictions_one_hot,np.reshape(classifier_scores,(-1,1)))
		total_votes=np.sum(weighted_votes,axis=0)
		label = np.argmax(total_votes,axis=(0))
		true_label=np.argmax(testY[i])
		if label==true_label:
			correct+=1
		
		# print(correct)

	accuracy=float(correct)/tot_test
	print("Prediction rule 5")
	print("Total test accuracy is :")
	print(accuracy)
	return accuracy

####Oracle baseline : The oracle is correct if any of the classifiers in the ensemble correctly guess the label
def train_and_test_oracle(ens,num_classifiers,num_classes,trainX,trainY,testX,testY):
	#predict
	tot_test=testX.shape[0]
	correct=0

	for i in range(0,tot_test):
		sample=testX[i]
		DP= ens.get_profile(sample)
		predictions = np.argmax(DP,axis=1)
		# print(predictions)
		# break
		true_label=np.argmax(testY[i])
		# print(true_label)
		# print(predictions==true_label)
		predictions=predictions==true_label
		predictions=predictions.astype(int)
		if np.max(predictions)==1:
			correct+=1
			# print(correct)


	accuracy=float(correct)/float(tot_test)
	print("Oracle :")
	print("Total test accuracy is :")
	print(accuracy)
	return accuracy

####Single best classifier baseline : Returns the accuracy of the single best classifier in the ensemble
def train_and_test_single_best(ens,num_classifiers,num_classes,trainX,trainY,testX,testY):
	#predict
	tot_test=testX.shape[0]
	correct=np.zeros(num_classifiers)

	for i in range(0,tot_test):
		sample=testX[i]
		DP= ens.get_profile(sample)
		predictions = np.argmax(DP,axis=1)
		# print(predictions)
		# break
		true_label=np.argmax(testY[i])
		# print(predictions==true_label)
		predictions=predictions==true_label
		predictions=predictions.astype(int)
		correct=correct + predictions

	winner=np.argmax(correct)
	max_correct=correct[winner]
	accuracy=max_correct/tot_test
	print("Single best classifier is classifier "+str(winner))
	print("Total test accuracy is :")
	print(accuracy)
	return accuracy