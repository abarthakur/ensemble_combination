import numpy as np
import matplotlib.pyplot as plt
import pandas

size1 = 2000
size1_test = 500
size2 = 2000
size2_test = 500

mean = [0.2, 0.2]
cov = [[0.01, 0.9], [0.9, 0.01]]
x1, y1 = np.random.multivariate_normal(mean, cov, size1).T
x1_test, y1_test = np.random.multivariate_normal(mean, cov, size1_test).T

mean = [1.5, 1.5]
cov = [[0.01, 0.9], [0.9, 0.01]]
x2, y2 = np.random.multivariate_normal(mean, cov, size2).T
x2_test, y2_test = np.random.multivariate_normal(mean, cov, size2_test).T

label0 = np.zeros(len(x1), dtype=np.int)
label1 = np.ones(len(x2), dtype=np.int)
label0_test = np.zeros(len(x1_test), dtype=np.int)
label1_test = np.ones(len(x2_test), dtype=np.int)

# plt.plot(x1, y1, 'ro')
# plt.plot(x2, y2, 'go')
# plt.show()

# train data
x1 = np.array(x1).reshape((x1.shape[0], 1)) # 1000x1
y1 = np.array(y1).reshape((1, y1.shape[0])) # 1x1000
label0 = label0.reshape((1, label0.shape[0]))
X1 = np.concatenate((x1,y1.T), axis=1) # 1000x2
X1 = np.concatenate((X1,label0.T), axis=1) # 1000x3

x2 = np.array(x2).reshape((x2.shape[0], 1)) # 1000x1
y2 = np.array(y2).reshape((1, y2.shape[0])) # 1x1000
label1 = label1.reshape((1, label1.shape[0]))
X2 = np.concatenate((x2,y2.T), axis=1) # 1000x2
X2 = np.concatenate((X2,label1.T), axis=1) # 1000x3

X = np.concatenate((X1, X2), axis=0)
np.random.shuffle(X)
pandas.DataFrame(X).to_csv("inseparable.trn", header=None, index=False)

# test data
x1_test = np.array(x1_test).reshape((x1_test.shape[0], 1)) # 1000x1
y1_test = np.array(y1_test).reshape((1, y1_test.shape[0])) # 1x1000
label0_test = label0_test.reshape((1, label0_test.shape[0]))
X1_test = np.concatenate((x1_test,y1_test.T), axis=1) # 1000x2
X1_test = np.concatenate((X1_test,label0_test.T), axis=1) # 1000x3

x2_test = np.array(x2_test).reshape((x2_test.shape[0], 1)) # 1000x1
y2_test = np.array(y2_test).reshape((1, y2_test.shape[0])) # 1x1000
label1_test = label1_test.reshape((1, label1_test.shape[0]))
X2_test = np.concatenate((x2_test,y2_test.T), axis=1) # 1000x2
X2_test = np.concatenate((X2_test,label1_test.T), axis=1) # 1000x3

X = np.concatenate((X1_test, X2_test), axis=0)
np.random.shuffle(X)
pandas.DataFrame(X).to_csv("inseparable.tst", header=None, index=False)