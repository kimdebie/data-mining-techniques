import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

def benchmark(data):

    '''Implements a benchmark metric for predicting mood: mood is the same as the previous day'''
    print(len(data))
    data = data[["id", "time", "mood"]]

    data["mood"].hist()
    #plt.show()

    # predicted mood is mood of previous day
    data["predicted_mood"] = data["mood"].shift(1)

    # we can only do this for observations with 1 day in between
    data = data[data["time"].diff() == 24].dropna()

    rsquared = r2_score(data["mood"], data["predicted_mood"])
    mse = mean_squared_error(data["mood"], data["predicted_mood"])

    ypred = data["predicted_mood"]
    print(len(ypred.values))
    Y_test = data["mood"]

    # Define accuracy with 10% error range
    accuracy = []
    for i in range(len(ypred)):
    	if ypred.values[i] < Y_test.values[i]*1.05 and ypred.values[i] > Y_test.values[i]*0.95:
    		accuracy.append(1)
    	else:
    		accuracy.append(0)
    acc = float(sum(accuracy)) / float(len(ypred.values))

    plt.scatter(data['mood'], data['predicted_mood'])
    #plt.show()

    return mse, acc, accuracy

# print('rsquared: %.2f' % rsquared)
# print('mse: %.5f' % mse) # this is low because data is downscaled (we should multiply it by 10 I think)
