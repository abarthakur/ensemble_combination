from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

from ensemble import *
from data import *

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

def plot_tsne_decision_space(ens,X,Y):
	num_samples=X.shape[0]
	num_classifiers=ens.num_classifiers
	num_classes=ens.num_classes
	DP_X= np.zeros((num_samples,num_classifiers*num_classes))
	true_labels=np.zeros(num_samples)
	for i in range(0,num_samples):
		DP_X[i,:]=ens.get_profile(X[i,:]).reshape(num_classifiers*num_classes)
		# print(DP_X[i,:])
		true_labels[i]=np.argmax(Y[i,:])

	print("Starting tSNE visualization of DP space")
	DP_X_embedded=TSNE(n_components=2).fit_transform(DP_X)
	x_0 = DP_X_embedded[:,0]
	x_1 = DP_X_embedded[:,1]
	plt.scatter(x_0,x_1,c=true_labels)
	plt.show()


def plot_tsne_classifier_regions(ens,X,Y):
	num_samples=X.shape[0]
	num_classifiers=ens.num_classifiers
	num_classes=ens.num_classes
	DP_X= np.zeros((num_samples,num_classifiers*num_classes))
	true_labels=np.zeros(num_samples)
	classifier_perf=np.zeros((num_samples,num_classifiers))

	for i in range(0,num_samples):
		DP=ens.get_profile(X[i,:])
		DP_X[i,:]=DP.reshape(num_classifiers*num_classes)
		true_label=np.argmax(Y[i,:])
		predictions = np.argmax(DP,axis=1)
		perf=predictions==true_label
		perf=perf.astype(int)
		classifier_perf[i,:]=perf		
		# print(DP_X[i,:])
		true_labels[i]=true_label

	print("Starting tSNE embedding of DP space")
	DP_X_embedded=TSNE(n_components=2).fit_transform(DP_X)
	x_0 = DP_X_embedded[:,0]
	x_1 = DP_X_embedded[:,1]
	plt.figure(0)
	plt.scatter(x_0,x_1,c=true_labels)
	plt.show()

	for i in range(0,num_classifiers):
		plt.figure(i)
		plt.title("Classifier "+str(i))
		plt.scatter(x_0,x_1,c=classifier_perf[:,i])
		plt.show()

def plot_tsne_classifier_regions_2(ens,X,Y,k_val):
	num_samples=X.shape[0]
	num_classifiers=ens.num_classifiers
	num_classes=ens.num_classes

	DP_X_flat= np.zeros((num_samples,num_classifiers*num_classes))
	true_labels=np.zeros(num_samples)

	for i in range(0,num_samples):
		DP=ens.get_profile(X[i,:])
		DP_X_flat[i,:]=DP.reshape(num_classifiers*num_classes)
		true_labels[i]=np.argmax(Y[i,:])

	nbrs = NearestNeighbors(n_neighbors=k_val, algorithm='ball_tree').fit(DP_X_flat)
	best_classifier=np.zeros(num_samples)
	counter=0
	counter2=0
	for i in range(0,num_samples):
		[d,indices]= nbrs.kneighbors(DP_X_flat[i,:].reshape(1,-1))
		# print(indices)
		scores=np.zeros(num_classifiers)
		confidence=np.zeros(num_classifiers)
		for j in indices[0]:
			# print(j)
			DP_nbr = DP_X_flat[j,:].reshape((num_classifiers,num_classes))
			# break
			predictions=np.argmax(DP_nbr,axis=1)
			new_scores=(predictions==true_labels[j]).astype(int)
			new_conf=DP_nbr[:,true_labels[j].astype(int)]
			new_conf=new_conf*new_scores
			confidence=confidence+new_conf
			scores=scores+new_scores
		
		winners=np.argwhere(scores == np.amax(scores))
		winners=winners.flatten().tolist()
		if len(winners)==1:
			best_classifier[i]=np.argmax(scores)
		else:
			counter+=1
			most_confident=np.argwhere(confidence == np.amax(confidence)).flatten().tolist()
			most_conf_winners=list(set(most_confident) & set(winners))
			if len(most_conf_winners)==0:
				print("Oh Snap! No intersection")
				counter2+=1
				best_classifier[i]=np.argmax(scores)	
			elif len(most_conf_winners)==1:
				best_classifier[i]=most_conf_winners[0]
			else:
				print("Oh Snap! Too large an intersection")
				counter2+=1
				best_classifier[i]=most_conf_winners[0]

	print("Number of instances with multiple winners: "+str(counter)+" "+str(counter2))

	print("Starting tSNE embedding of DP space")
	DP_X_embedded=TSNE(n_components=2).fit_transform(DP_X_flat)
	x_0 = DP_X_embedded[:,0]
	x_1 = DP_X_embedded[:,1]

	fig, axes = plt.subplots(nrows=1, ncols=2)
	axes[0].scatter(x_0,x_1,s=8,c=true_labels,alpha=0.5)
	axes[0].set_title("True Class Labels")
	axes[1].scatter(x_0,x_1,s=8,c=best_classifier,alpha=0.5)
	axes[1].set_title("Best Classifier")
	plt.show()

def plot_tsne_locality(ens,X,Y,k_for_means):
	num_samples=X.shape[0]
	num_classifiers=ens.num_classifiers
	num_classes=ens.num_classes

	DP_X_flat= np.zeros((num_samples,num_classifiers*num_classes))
	true_labels=np.zeros(num_samples)

	for i in range(0,num_samples):
		DP=ens.get_profile(X[i,:])
		DP_X_flat[i,:]=DP.reshape(num_classifiers*num_classes)
		true_labels[i]=np.argmax(Y[i,:])

	#cluster DP_X_flat
	kmeans = KMeans(n_clusters=k_for_means, random_state=0).fit(DP_X_flat)
	labels=kmeans.labels_
	X_embedded=TSNE(n_components=2).fit_transform(X)
	x_0 = X_embedded[:,0]
	x_1 = X_embedded[:,1]
	fig, axes = plt.subplots(nrows=1, ncols=2)
	axes[0].scatter(x_0,x_1,s=8,c=true_labels,alpha=0.5)
	axes[0].set_title("True Class Labels")
	axes[0].legend()
	axes[1].scatter(x_0,x_1,s=8,c=labels,alpha=0.5)
	axes[1].set_title("Cluster Labels in DP Space")
	# axes[1].legend()
	plt.show()


# trainX, trainY = get_sat_data("data/sat.trn")
# testX, testY = get_sat_data("data/sat.tst")
# num_classes = trainY.shape[1]
# permutation = np.random.permutation(trainX.shape[0])
# trainX=trainX[permutation[0:100],:]
# trainY=trainY[permutation[0:100],:]
# # print(trainX.shape)
# # quit()
# ens = build_ensemble_kuncheva_sat(trainX,trainY,num_classes)
# num_classifiers=ens.num_classifiers

# # plot_tsne_classifier_regions_2(ens,trainX,trainY,10)
# plot_tsne_locality(ens,trainX,trainY,15)