from income.util import getFilteredData
import numpy as np
import pymc3 as pm
import theano
floatX = theano.config.floatX

def setup(training_data="../data/income/2014.csv", test_data="../data/income/2017.csv", sample_size=-1):
    if isinstance(training_data, list):
        for training in training_data:
            data = getFilteredData(training).head(30000)
            if (train_data != None):
                train_data.append(data)
            else:
                train_data = data
    else:
        train_data = getFilteredData(training_data).head(30000)
    test_data = getFilteredData(test_data)

    if (sample_size != -1):
        train_data = train_data.sample(sample_size).dropna(axis=1)

    column_list=['agi_stub','year','zipcode','avg_dep']
    label=['avg_total_income']

    X_train = train_data.loc[:,column_list]
    Y_train = train_data.loc[:,label]
    X_test = test_data.loc[:,column_list]
    Y_test = test_data.loc[:,label]

    return X_train, Y_train, X_test, Y_test

def construct_nn(ann_input, ann_output):
    n_hidden = 5

    numFeatures = ann_input.get_value().shape[1];

    init_in_1 = np.random.randn(numFeatures, n_hidden).astype(floatX)
    init_1_2 = np.random.randn(n_hidden, n_hidden).astype(floatX)
    init_2_out = np.random.randn(n_hidden).astype(floatX)

    with pm.Model() as neural_network:
        weights_in_1 = pm.Normal('w_in_1', 0, sd=1,
                                 shape=(numFeatures, n_hidden), testval=init_in_1)
        weights_1_2 = pm.Normal('w_1_2', 0, sd=1,
                                shape=(n_hidden, n_hidden), testval=init_1_2)
        weights_2_out = pm.Normal('w_2_out', 0, sd=1,
                                  shape=(n_hidden,), testval=init_2_out)

        act_1 = pm.math.tanh(pm.math.dot(ann_input, weights_in_1))
        act_2 = pm.math.tanh(pm.math.dot(act_1, weights_1_2))
        act_3 = pm.math.sigmoid(pm.math.dot(act_2, weights_2_out))

        a = pm.Uniform('uni_1', lower=0, upper=5000)
        act_out = pm.Deterministic('det_1', a * act_3)
        act_out = pm.HalfNormal('half_norm_1', a * act_out, observed=ann_output)

        out = pm.Normal('out', act_out, observed=ann_output)

    return neural_network