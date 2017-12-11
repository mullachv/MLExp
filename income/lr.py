import numpy as np
import matplotlib.pyplot as plt
from income.setup import setup

X, Y = setup()

X = X["avg_dep"]

denominator = X.dot(X) - X.mean() * X.sum()
a = ( X.dot(Y) - Y.mean() * X.sum() ) / denominator
b = ( Y.mean() * X.dot(X) - X.mean() * X.dot(Y) ) / denominator

Yhat = a*X + b

plt.scatter(X, Y)
plt.plot(X, Yhat)
plt.show()

d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)
print("a: ", a, "b: ", b)
print("the r-squared is:", r2)



