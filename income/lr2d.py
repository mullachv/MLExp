import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from income.setup import setup

X, Y = setup()
X = X[['zipcode','avg_dep']]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X.iloc[:,0],X.iloc[:,1],Y)
plt.show()

w=np.linalg.solve(np.dot(X.T,X), np.dot(X.T,Y))
Yhat = np.dot(X,w)

d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)
print("the r-squared is:", r2)



