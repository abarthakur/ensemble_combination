from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

from ensemble import *
from data import *

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



# trainX, trainY = get_sat_data("data/sat.trn")
# testX, testY = get_sat_data("data/sat.tst")
# num_classes = trainY.shape[1]

# ens = build_ensemble_kuncheva_sat(trainX,trainY,num_classes)
# num_classifiers=ens.num_classifiers

# plot_tsne_decision_space(ens,trainX,trainY)
