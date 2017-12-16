import pandas
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def getScaler(data, columns, scalers):
    if str(columns) in scalers:
        return scalers[str(columns)]
    else:
        scalers[str(columns)] = MinMaxScaler(copy=True, feature_range=(0, 1))
        data[columns] = scalers[str(columns)].fit_transform(data[columns])
    return scalers[str(columns)]

def getFilteredData(filename):
    data = pandas.read_csv(filename)
    data.columns = data.columns.str.lower()
    # relevant zip code, has returns, has income
    data = data[(data["zipcode"] > 501) & (data["zipcode"] < 99951) & (data['n1'] > 0) & (data['a02650'] > 0) & (data['n1'] >= data['mars1'])]
    data["year"] = int(filename.split('/')[len(filename.split('/'))-1][0:4])
    data = data[data['state'] == 'NY'].loc[:,
            [
                "n1", # num returns
                "agi_stub", # 1,2,3,4,5,6
                "year",
                "zipcode",
                #"state",
                "a00100", #agi
                "a04800", #taxable income,
                "a04800", #num ret w/ taxable income
                "a02650", #total income
                "n02650", #num ret w/ income
                "numdep",
                "mars1",  #single
                "mars2",  #joint
                "mars4"   #head of household
            ]
        ]
    data["avg_dep"] = round(data["numdep"]/(data["n1"] - data["mars1"]))
    data["avg_dep"] = data["avg_dep"].replace([np.inf, np.nan], 0)
    data["avg_total_income"] = data["a02650"]/data["n02650"]
    #print(data.head(1))
    data.rename(columns={"n1":"numret","a00100":"agi","a04800":"taxable_income"}, inplace=True)
    return data
