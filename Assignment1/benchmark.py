import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

def benchmark(file):

    '''Implements a benchmark metric for predicting mood: mood is the same as the previous day'''

    data = pd.read_csv(file)

    data = data[["id", "time", "mood"]]

    data["mood"].hist()
    plt.show()

    # predicted mood is mood of previous day
    data["predicted_mood"] = data["mood"].shift(1)

    # we can only do this for observations with 1 day in between
    data = data[data["time"].diff() == 24].dropna()

    data["accuratehighbound"] = data["mood"] < data["predicted_mood"] * 1.05
    data["accuratelowbound"] =  data["mood"] > data["predicted_mood"] * 0.95
    data["accurate"] = data[["accuratelowbound", "accuratehighbound"]].all(axis='columns')

    print(data.head())

    accuracy = data["accurate"].sum() / data["accurate"].count()
    print(accuracy)

    rsquared = r2_score(data["mood"], data["predicted_mood"])
    mse = mean_squared_error(data["mood"], data["predicted_mood"])

    plt.scatter(data['mood'], data['predicted_mood'], c=data["id"])
    plt.show()

    return rsquared, mse


rsquared, mse = benchmark('cleaned_normalized.csv')

print('rsquared: %.2f' % rsquared)
print('mse: %.5f' % mse) # this is low because data is downscaled (we should multiply it by 10 I think)
