import preprocess
import analyze
import pivot
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, accuracy_score


data = pd.read_csv('with_features.csv', index_col=0)
# data = data.drop(columns=["Unnamed: 0"])
data = data.drop(columns=["time"])

msk = np.random.rand(len(data)) < 0.8
train_data = data[msk]
test_data = data[~msk]

# Define train and test data 
X_train = train_data.drop(columns = ['mood', 'id']) #, 'circumplex.arousal_lag1', 'circumplex.arousal_lag2', 'circumplex.arousal_lag3', 'circumplex.valence_lag1', 'circumplex.valence_lag2', 'circumplex.valence_lag3', 'mood_lag1', 'mood_lag2', 'mood_lag3', 'mood_lag4', 'circumplex.arousal', 'circumplex.valence'])
Y_train = train_data['mood']
X_test = test_data.drop(columns = ['mood', 'id']) #, 'circumplex.arousal_lag1', 'circumplex.arousal_lag2', 'circumplex.arousal_lag3', 'circumplex.valence_lag1', 'circumplex.valence_lag2', 'circumplex.valence_lag3', 'mood_lag1', 'mood_lag2', 'mood_lag3', 'mood_lag4', 'circumplex.arousal', 'circumplex.valence'])
Y_test = test_data['mood']


# Fit and summarize OLS model
mod = sm.OLS(Y_train, X_train)

res = mod.fit()

ypred = res.predict(X_test)

# print(res.summary())
print("RMSE:")
print(mean_squared_error(Y_test, ypred))

# Define accuracy with 10% error range
accuracy = []
for i in range(len(ypred)):
	if ypred.values[i] < Y_test.values[i]*1.05 and ypred.values[i] > Y_test.values[i]*0.95:
		accuracy.append(1)
	else:
		accuracy.append(0)

print("Accuracy")
print(float(sum(accuracy)) / float(len(ypred.values)))