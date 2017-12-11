import numpy as np
import matplotlib.pyplot as plt
from income.setup import setup
from income.util import getScaler

X, Y = setup(sample_size=100)
columns = ['agi_stub','zipcode']
X = X[columns]

scalers = {}
scaler1 = getScaler(X, columns, scalers)

w = np.random.randn(X.shape[1])

z = X.dot(w)
b = 0

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def forward(X, W, b):
    return sigmoid(X.dot(W) + b)

P_Y_given_X = forward(X, w, b)
predictions = np.round(P_Y_given_X)

def classification_rate(Y, P):
    return np.mean(Y == P)

print ("Score:", classification_rate(Y, predictions))