"""
SVM with linear kernel. Dataset is averaged in order to perform classification.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

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
def average_k_dataset(data, k, target_row=False):
    # initialize dataframes
    mean_df = pd.DataFrame()
    previous_k = pd.DataFrame()
    targets = []

    # iterate over data rows
    for idx, row in data.iterrows():
        previous_k = previous_k.append(row)

        # check if current row is used as target
        if target_row == True:
            target_row = False
            continue

        # after k rows, calculate mean and set target (if end is not reached!)
        if (idx+1) % k == 0 and idx != (data.shape[0]-1):
            mean_previous = previous_k.mean(axis=0)
            mean_df = mean_df.append(mean_previous, ignore_index=True)
            targets.append(data.iloc[idx+1]['mood'])
            previous_k = pd.DataFrame()
            target_row = True

    #df = target_to_label(mean_df, targets, 3)
    df = balanced_classes(mean_df, targets, 4)
    return df

"""
Convert mood targets to labels and append to dataset
"""
def target_to_label(df, targets, n_classes=3):
    # positive, neutral, negative mood class
    if n_classes == 3:
        labels = []
        for t in targets:
            if t<=0.33:
                labels.append(1)
            elif t>0.33 and t<=0.66:
                labels.append(2)
            else:
                labels.append(3)

    # scale of 1 to 10
    elif n_classes == 10:
        labels = ["{0:.1f}".format(t) for t in targets]
    else:
        raise Exception("Number of classes must be 3 or 10")

    df['label'] = labels
    return df

def balanced_classes(df, targets, n_classes):
    labels = np.zeros((len(targets),), dtype=int)
    sorted_targets = np.sort(targets)
    arg_sorted_targets = np.argsort(targets)
    quantiles = np.array_split(sorted_targets, n_classes)

    # i represents the class label (1...n_classes)
    j=0
    for i, quantile in enumerate(quantiles):
        for target in quantile:
            idx = arg_sorted_targets[j]
            labels[idx] = i+1
            j+=1

    df['label'] = labels
    return df

def linear_SVM(data, n_classes):

    # read data
    # data = pd.read_csv('cleaned_normalized.csv', header = 0)
    # data = data.drop(columns=["Unnamed: 0"])

    # average k days, (k+1)'th day is target
    #avg_dataset = average_k_dataset(data, k)

    data = balanced_classes(data, data['mood'].as_matrix(), n_classes)
    print("Shape of dataset: {}".format(data.shape))
    print(data[data['label']==1]['mood'].shape)
    print(data[data['label']==2]['mood'].shape)
    print(data[data['label']==3]['mood'].shape)
    print(data[data['label']==4]['mood'].shape)

    # divide data in attributes and target
    # X = data.drop(['label', 'mood', 'circumplex.arousal', 'circumplex.valence'], axis=1)
    X = data.drop([['mood'], ['label']], axis=1)
    print(X.columns)
    y = data['label']

    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
    print("Shape of training data: {} \n Shape of test data: {}".format(X_train.shape, X_test.shape))

    # train classifier
    svclassifier = SVC(kernel='linear')
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)

    # evaluation metrics
    print("-------------------Confusion matrix-------------------")
    print(confusion_matrix(y_test,y_pred))
    print("-------------Precision and Recall metrics-------------")
    print(classification_report(y_test,y_pred))
    accuracy = accuracy_score(y_test,y_pred)
    print("-----------------------Accuracy-----------------------\n{0:.3f}".format(accuracy))
