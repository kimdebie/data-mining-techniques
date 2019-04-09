'''
Functions for preprocessing the data
'''

import pandas as pd
from datetime import datetime
from scipy import stats
import numpy as np
#import pyplot.matplotlib as plt

def load(filename):

    '''Load data from file into initial format'''

    # initial loading and formatting
    data = pd.read_csv(filename)
    data = data.drop(columns=["Unnamed: 0"])
    data["time"] = data["time"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f"))

    # save data on day level (change this line for different granularity)
    data["time"] = data["time"].apply(lambda x: x.date())

    # take mean values (aggregating at the specified datetime level)
    data = data.groupby(["id", "time", "variable"]).mean().reset_index()

    #print(data.head())

    # variables as columns
    pivoted = data.pivot_table(index=['id', 'time'], columns='variable', values='value').reset_index()

    print(pivoted.shape)

    return pivoted


def clean(data):

    '''Remove redundant rows, outliers, normalize data etc.'''

    # remove days for which there is no mood data
    # note: the list may be extended with other values that must be present
    data = data.dropna(subset=['mood'])

    # instead of exact datetime, calculate datetime relative to first datetime (in miliseconds)
    data['time'] = data.groupby('id')['time'].transform(lambda x: x - x.min())


    print(data.head())

    print(data.shape)

    data.to_csv('cleaned.csv')

    return data


def create_features_ML(clean_data):

    raise NotImplementedError

    return featurized_data


def create_features_temporal(clean_data):

    raise NotImplementedError

    return featurized_data
