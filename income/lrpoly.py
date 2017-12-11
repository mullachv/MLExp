import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from income.setup import setup

X, Y = setup(100)

X2 = []
for x in X["avg_dep"]:
    X2.append([1, x, x*x])

X = np.array(X2)

# plt.scatter(X[:, 1], Y)
# plt.show()

w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
Yhat = np.dot(X,w)
plt.scatter(X[:, 1], Y)
plt.plot(sorted(X[:, 1]),sorted(Yhat))
plt.show()

Yhat = X.dot(w)
d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)
print("the r-squared is:", r2)
