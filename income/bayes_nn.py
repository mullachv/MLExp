from income.setup import setup, construct_nn
import theano
import numpy as np
import pymc3 as pm

X_train, Y_train, X_test, Y_test = setup(training_data="../data/income/2014.csv",test_data="../data/income/2015.csv")

ann_input = theano.shared(np.array(X_train))
ann_output = theano.shared(np.array(Y_train['avg_total_income']))
neural_network = construct_nn(ann_input, ann_output)

with neural_network:
    inference = pm.ADVI()
    approx = pm.fit(n=3000, method=inference)

ann_input.set_value(X_test)
ann_output.set_value(Y_test['avg_total_income'])
trace = approx.sample(draws=500)

with neural_network:
    ppc = pm.sample_ppc(trace, samples=500, progressbar=True)


approx