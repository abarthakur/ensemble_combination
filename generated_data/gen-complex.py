import numpy as np
import matplotlib.pyplot as plt
import pandas

# eqn: y = sin(3x)*3

size1 = 1000
size1_test = 250
size2 = 1000
size2_test = 250

unif_points = np.random.uniform(-5,5,size=(size1+size2+size1_test+size2_test,2))
classified_points = np.zeros((unif_points.shape[0], 3)) # 2d points and label

x1 = []
y1 = []
x2 = []
y2 = []
for i in range(unif_points.shape[0]):
    val = unif_points[i]
    x = val[0]
    y = val[1]
    label = 0
    if y > np.sin(3*x)*3:
        label = 1
        x1.append(x)
        y1.append(y)
    else:
        x2.append(x)
        y2.append(y)

    classified_points[i][0] = x
    classified_points[i][1] = y
    classified_points[i][2] = label


plt.plot(x1, y1, 'ro')
plt.plot(x2, y2, 'go')
plt.show()

pandas.DataFrame(classified_points[0:size1+size2]).to_csv("sine.trn", header=None, index=False)
pandas.DataFrame(classified_points[size1+size2:classified_points.shape[0]]).to_csv("sine.tst", header=None, index=False)

