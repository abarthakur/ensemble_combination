import numpy as np
import matplotlib.pyplot as plt
import pandas

from keras.models import Sequential
from keras.layers import Dense

def get_data(filename): # Satimage dataset
    data = pandas.read_csv(filename, sep=r"\s+", header=None)
    data = data.values

    dataX = np.array(data[:,range(data.shape[1]-1)])
    dataY = np.array(data[np.arange(data.shape[0]),data.shape[1]-1])

    # convert dataY to one-hot, 6 classes
    num_classes = 6
    dataY = np.array([x-2 if x==7 else x-1 for x in dataY]) # re-named class 7 to 6(5)
    dataY_onehot = np.zeros([dataY.shape[0], num_classes])
    dataY_onehot[np.arange(dataY_onehot.shape[0]), dataY] = 1

    return dataX, dataY_onehot




trainX, trainY = get_data("data/sat.trn")
testX, testY = get_data("data/sat.tst")
num_classes = trainY.shape[1]


#### Create Models ####

model1 = Sequential()
model1.add(Dense(8, input_dim=trainX.shape[1], activation="sigmoid"))
model1.add(Dense(num_classes, activation="softmax"))
model1.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
model1.fit(trainX, trainY, epochs=2, batch_size=10) # epochs=150
# scores = model1.evaluate(testX, testY)
# print("\n%s: %.2f%%" % (model1.metrics_names[1], scores[1]*100))

model2 = Sequential()
model2.add(Dense(8, input_dim=trainX.shape[1], activation="sigmoid"))
model2.add(Dense(8, activation="sigmoid"))
model2.add(Dense(num_classes, activation="softmax"))
model2.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
model2.fit(trainX, trainY, epochs=2, batch_size=10) # epochs=150


#### Decision Templates ####

num_classifiers = 2

DT = np.zeros([num_classes, num_classifiers, num_classes])
counts = np.zeros(num_classes)

for i in range(trainX.shape[0]):
    c = 0 # class to which trainX[i] belongs
    for j in range(trainY[i].shape[0]):
        if trainY[i,j]==1:
            c = j
    DT[c] = DT[c] + np.array( [model1.predict(np.array([trainX[i]])).reshape(num_classes), model2.predict(np.array([trainX[i]])).reshape(num_classes)] )
    counts[c] += 1

for c in range(DT.shape[0]):
    DT[c] = np.divide(DT[c], counts[c])

# DT[i] is the Decision Template for class i

print(DT)

#classify



#### k nearest neighbours ####

## Decision Profile
# DP = np.array( [model1.predict(np.array([testX[i]])).reshape(num_classes), model2.predict(np.array([testX[i]])).reshape(num_classes)] )





