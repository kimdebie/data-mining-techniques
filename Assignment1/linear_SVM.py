"""
SVM with linear kernel. Dataset is averaged in order to perform classification.
"""

import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

"""
---USER SPECIFIC---
Take average of variables per user
"""
def average_dataset(data):
    mean_df = pd.DataFrame()
    unique_ids = data['id'].unique()
    targets = []

    for k, id in enumerate(unique_ids):
        rows = data.loc[data['id'] == id]

        # average all variables over the time period
        mean = rows.mean(axis=1)

        # set last time point as target
        last_day = rows['time'].max()
        y = rows.loc[rows['time'] == last_day]
        label = y['mood']

        # append to dataframe
        mean_df = mean_df.append(mean, ignore_index=True)
        targets.append(label.iloc[0])

    df = target_to_label(mean_df, targets)

    return df

"""
---GENERAL---
Average previous k days and set (k+1)'th day as target
"""
def average_k_dataset(data, k):
    # initialize dataframes
    mean_df = pd.DataFrame()
    previous_k = pd.DataFrame()
    targets = []
    target_row = False

    # iterate over data rows
    for idx, row in data.iterrows():

        # check if current row is used as target
        if target_row == True:
            target_row = False
            continue

        previous_k = previous_k.append(row)
        
        # after k rows, calculate mean and set target (if end is not reached!)
        if (idx+1) % k == 0 and idx != (data.shape[0]-1):
            print(previous_k)
            mean_previous = previous_k.mean(axis=1,skipna=True)
            mean_df = mean_df.append(mean_previous, ignore_index=True)
            targets.append(data.iloc[idx+1]['mood'])
            previous_k = pd.DataFrame()
            target_row = True

    df = target_to_label(mean_df, targets)
    return df

"""
Convert mood targets to labels and append to dataset
"""
def target_to_label(df, targets):
    labels = ["{0:.1f}".format(t) for t in targets]
    df['label'] = labels
    return df

# read data
data = pd.read_csv('cleaned_normalized.csv', header = 0)
data = data.drop(columns=["Unnamed: 0"])

# average 4 days, 5th day is target
av_dataset = average_k_dataset(data, 4)
print(av_dataset.head())

# get average data set and mood class
# dataset = average_dataset(data)

# divide data in attributes and target
# X = av_dataset.drop('label', axis=1)
# y = av_dataset['label']
#
# # split data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
# print("Shape of training data: {} \n Shape of test data: {}".format(X_train.shape, X_test.shape))
#
# # train classifier
# svclassifier = SVC(kernel='linear')
# svclassifier.fit(X_train, y_train)
# y_pred = svclassifier.predict(X_test)
#
# print(confusion_matrix(y_test,y_pred))
# print(classification_report(y_test,y_pred))