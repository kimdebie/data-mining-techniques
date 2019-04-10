'''
Functions for preprocessing the data
'''

import pandas as pd
from datetime import datetime
from scipy import stats
import numpy as np
from sklearn import preprocessing
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

def load(filename):

    '''Load data from file into initial format'''

    # initial loading and formatting
    data = pd.read_csv(filename)
    data = data.drop(columns=["Unnamed: 0"])
    data["time"] = data["time"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f"))

    # save data on day level (change this line for different granularity)
    data["time"] = data["time"].apply(lambda x: x.replace(hour=0,minute=0,second=0,microsecond=0).timestamp()/ (60*60))

    # remove outliers (technically cleaning but hard to do later!)
    no_outliers = data.groupby(["id", "variable"])["value"].transform(lambda group: no_outlier(group)).to_frame()
    data = data[no_outliers.any(axis=1)]

    # variables as columns
    data = data.pivot_table(index=['id', 'time'], columns='variable', values='value').reset_index()

    # aggregating methods for each column (mean or sum) depending on data semantics
    aggtypes = {'mood': 'mean', 'circumplex.arousal': 'mean', 'circumplex.valence': 'mean',
                'appCat.builtin': 'sum', 'appCat.communication': 'sum', 'appCat.entertainment': 'mean',
                'appCat.finance': 'sum', 'appCat.game': 'sum', 'appCat.office': 'sum', 'appCat.other': 'sum',
                'appCat.social': 'sum', 'appCat.travel': 'sum', 'appCat.unknown': 'sum',
                'appCat.utilities': 'sum', 'appCat.weather': 'sum', 'call': 'sum', 'screen': 'sum', 'sms': 'sum',
                'activity': 'mean'}

    # apply aggregations (at the specified datetime level)
    pivoted = data.groupby(["id", "time"]).agg(aggtypes).reset_index()

    print(pivoted.shape)
    print(pivoted.head())

    return pivoted


def clean(data):

    '''Remove redundant rows, outliers, normalize data etc.'''

    # remove days for which there is no mood data
    # note: the list may be extended with other values that must be present
    data = data.dropna(subset=['mood'])

    # instead of exact datetime, calculate datetime relative to first datetime (timediff in hours)
    data['time'] = data.groupby('id')['time'].transform(lambda x: x - x.min())

    print(data.head())
    print(data.shape)

    # select columns that need to be normalized
    selected_df = data.iloc[:,2:]

    # replace NaN with most frequent value --> WE CAN ALSO USE ANOTHER METHOD THAT DEALS WITH NANs
    imp = SimpleImputer(strategy="most_frequent")
    cleaned_df = imp.fit_transform(selected_df)
    cleaned_df = pd.DataFrame(cleaned_df, columns = selected_df.columns)

    # perform normalization
    min_max_scaler = preprocessing.MinMaxScaler()
    data_normalized = pd.DataFrame(min_max_scaler.fit_transform(cleaned_df), columns = cleaned_df.columns, index = cleaned_df.index)

    # verification - plot few observations
    print(data_normalized.head())
    data_normalized['mood'].iloc[1:10].plot(kind='bar')
    plt.show()

    # https://scikit-learn.org/stable/modules/preprocessing.html
    # super useful library for preprocessing data!

    # https://chrisalbon.com/python/data_wrangling/pandas_normalize_column/
    # uses that library - seems a good idea!

    # some other blogs i came across:
    # https://www.analyticsvidhya.com/blog/2016/07/practical-guide-data-preprocessing-python-scikit-learn/
    # https://medium.com/@sidereal/feature-preprocessing-for-machine-learning-2f165d12012a

    # get id and time columns
    first_columns = data.iloc[:,:2]

    # write to csv
    data = pd.concat([first_columns.reset_index(drop=True), data_normalized.reset_index(drop=True)], axis = 1)
    print(data.head())
    data.to_csv('cleaned_normalized.csv')

    return data

def create_features_ML(clean_data):

    raise NotImplementedError

    return featurized_data


def create_features_temporal(clean_data):

    raise NotImplementedError

    return featurized_data

def no_outlier(x, z=3):

    '''Detect outliers by calculating z-score. A value is an outlier if it is
    more than z z-scores out'''

    zscores = (x - x.mean()).div(x.std() + 1e-10)

    return zscores.apply(lambda x: False if pd.notnull(x) and x > z else True)
