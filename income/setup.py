from income.util import getFilteredData
import numpy as np

def setup(training_data="../data/income/2014.csv", test_data="../data/income/2017.csv", sample_size=-1):
    train_data = getFilteredData(training_data).head(30000)
    test_data = getFilteredData(test_data)

    if (sample_size != -1):
        train_data = train_data.sample(sample_size)

    column_list=['agi_stub','year','zipcode','avg_dep']
    label='avg_total_income'

    X_train = train_data.loc[:,column_list]
    Y_train = train_data.loc[:,label]
    X_test = test_data.loc[:,column_list]
    Y_test = test_data.loc[:,label]

    return X_train, Y_train