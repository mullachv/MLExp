import matplotlib.pyplot as plt
from income.setup import setup
import pymc3 as pm

X, Y = setup()

inference = pm.ADVI()
approx = pm.fit(n=30000, method=inference)
