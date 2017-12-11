import pandas
from sklearn.ensemble import AdaBoostClassifier  # For Classification
from sklearn.metrics import zero_one_loss
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

from income.util import getFilteredData

train_data = getFilteredData("data/income/2008.csv").head(30000);
test_data = getFilteredData("data/income/2017.csv")


# scaler = MinMaxScaler()
# scaler.fit(train_data);
# train_data = pandas.DataFrame(scaler.transform(train_data));
#
# scaler = MinMaxScaler()
# scaler.fit(test_data)
# test_data = pandas.DataFrame(scaler.transform(test_data));

column_list=['year','zipcode']
label='taxable_income'

X_train = train_data.loc[:,column_list]
Y_train = train_data.loc[:,label]
X_test = test_data.loc[:,column_list]
Y_test = test_data.loc[:,label]

scalerX = MinMaxScaler().fit(X_train)
scalery = MinMaxScaler().fit(pandas.DataFrame(Y_train))
X_train = scalerX.transform(X_train)
y_train = scalery.transform(pandas.DataFrame(Y_train))
X_test = scalerX.transform(X_test)
y_test = scalery.transform(pandas.DataFrame(Y_test))

dt = DecisionTreeClassifier()
clf = AdaBoostClassifier(n_estimators=10, base_estimator=dt,learning_rate=1)
#Above I have used decision tree as a base estimator, you can use any ML learner as base estimator if it ac# cepts sample weight
clf.fit(X_train,Y_train)
y_pred = clf.predict(X_test)

y_pred = scalery.inverse_transform(pandas.DataFrame(y_pred))
Y_test = scalery.inverse_transform(pandas.DataFrame(Y_test))

for d in range(0, len(y_pred)):
    print(str(Y_test[d]) + " vs " + str(y_pred[d]))

accuracy = zero_one_loss(Y_test, y_pred)

print(accuracy)

