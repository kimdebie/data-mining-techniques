import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

def benchmark(data):

    '''Implements a benchmark metric for predicting mood: mood is the same as the previous day'''

    data = data[["id", "time", "mood", "mood_lag1"]]

    data["mood"].hist()
    #plt.show()

    data["accuratehighbound"] = data["mood"] < data["mood_lag1"] * 1.05
    data["accuratelowbound"] =  data["mood"] > data["mood_lag1"] * 0.95
    data["accurate"] = data[["accuratelowbound", "accuratehighbound"]].all(axis='columns')

    acc = data["accurate"].sum() / data["accurate"].count()

    accuracy = data["accurate"].tolist()

    rsquared = r2_score(data["mood"], data["mood_lag1"])
    mse = mean_squared_error(data["mood"], data["mood_lag1"])

    plt.scatter(data['mood'], data['mood_lag1'])
    #plt.show()

    return mse, acc, accuracy

# print('rsquared: %.2f' % rsquared)
# print('mse: %.5f' % mse) # this is low because data is downscaled (we should multiply it by 10 I think)
