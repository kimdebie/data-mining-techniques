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

def linearregr(X_train, X_test, Y_train, Y_test):


	# Fit and summarize OLS model
	mod = sm.OLS(Y_train, X_train)
	res = mod.fit()
	ypred = res.predict(X_test)

	# Define accuracy with 10% error range
	accuracy = []
	for i in range(len(ypred)):
		if ypred.values[i] < Y_test.values[i]*1.05 and ypred.values[i] > Y_test.values[i]*0.95:
			accuracy.append(1)
		else:
			accuracy.append(0)

	mse = mean_squared_error(Y_test, ypred)
	acc = float(sum(accuracy)) / float(len(ypred.values))

	return mse, acc, accuracy
