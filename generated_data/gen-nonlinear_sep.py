import numpy as np
import matplotlib.pyplot as plt
import pandas

size1 = 1000
size2 = 1000

mean = [0.2, 0.2]
cov = [[0.1, 0], [0, 0.1]]
x1, y1 = np.random.multivariate_normal(mean, cov, size1/2).T
mean = [1.4, 1.4]
cov = [[0.1, 0], [0, 0.1]]
x11, y11 = np.random.multivariate_normal(mean, cov, size1/2).T
x1 = np.concatenate((x1,x11))
y1 = np.concatenate((y1,y11))


mean = [1.4, 0.2]
cov = [[0.1, 0], [0, 0.1]]
x2, y2 = np.random.multivariate_normal(mean, cov, size2/2).T
mean = [0.2, 1.4]
cov = [[0.1, 0], [0, 0.1]]
x22, y22 = np.random.multivariate_normal(mean, cov, size2/2).T
x2 = np.concatenate((x2,x22))
y2 = np.concatenate((y2,y22))

label0 = np.zeros(len(x1), dtype=np.int)
label1 = np.ones(len(x2), dtype=np.int)

# plt.plot(x1, y1, 'ro')
# plt.plot(x2, y2, 'go')
# plt.show()

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
pandas.DataFrame(X).to_csv("well_separated_nonlinear.csv", header=None, index=False)
