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
    data["time"] = data["time"].apply(lambda x: x.replace(hour=0,minute=0,second=0,microsecond=0))

    # remove outliers (technically cleaning but hard to do later!)
    no_outliers = data.groupby(["id", "variable"])["value"].transform(lambda group: no_outlier(group)).to_frame()
    data = data[no_outliers.any(axis=1)]

    # variables as columns
    data = data.pivot_table(index=['id', 'time'], columns='variable', values='value').reset_index()
    print(data.head())

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


    return data

def weather_info(weather_file):

    ''''Load data from KNMI.'''

    weather_data = pd.read_csv(weather_file, skiprows = 15)
    weather_data = weather_data.drop(weather_data.columns[0], axis=1)

    headers = ['time', 'min_temp', 'max_temp', 'sun', 'rain']
    weather_data.columns = headers

    # set datetime object
    weather_data["time"] = weather_data["time"].astype(str)
    weather_data["time"] = weather_data["time"].apply(lambda x: datetime.strptime(x, "%Y%m%d"))
    weather_data["time"] = weather_data["time"].apply(lambda x: x.replace(hour=0,minute=0,second=0,microsecond=0).timestamp()/ (60*60))

    return weather_data


def clean(data):

    '''Remove redundant rows, outliers, normalize data etc.'''

    # include weekday
    data = weekday_dummies(data)
    data["time"] = data["time"].apply(lambda x: x.timestamp() / (60*60))


    # merge weather
    weather_data = weather_info('KNMI_20140701.csv')
    data = pd.merge(data, weather_data, how='left', on='time')

    # remove days for which there is no mood data
    # note: the list may be extended with other values that must be present
    data = data.dropna(subset=['mood'])


    # instead of exact datetime, calculate datetime relative to first datetime (timediff in hours)
    data['time'] = data.groupby('id')['time'].transform(lambda x: x - x.min())

    print(data.head())
    # print(data.shape)

    # select columns to be normalized
    nan_to_replace = [col for col in data.columns if not col in ['id', 'time']]

    # replace NaN with mean for that specific person (mode not suitable with sparse values)
    data[nan_to_replace] = data.groupby(['id'])[nan_to_replace].transform(lambda x: x.fillna(x.mean()))

    # create column for total appusage
    data['total_appuse'] = data['appCat.builtin'].fillna(0) + data['appCat.communication'].fillna(0) + data['appCat.entertainment'].fillna(0) \
         + data['appCat.finance'].fillna(0) + data['appCat.game'].fillna(0) + data['appCat.office'].fillna(0) + data['appCat.other'].fillna(0) \
         + data['appCat.social'].fillna(0) + data['appCat.travel'].fillna(0) + data['appCat.unknown'].fillna(0) + data['appCat.utilities'].fillna(0) \
         + data['appCat.weather'].fillna(0)

    cleaned_df = data

    print(cleaned_df.columns)

    # perform normalization on required columns
    columns_to_scale = [col for col in data.columns if not col in ['id', 'time']]
    min_max_scaler = preprocessing.MinMaxScaler()
    cleaned_df[columns_to_scale] = min_max_scaler.fit_transform(cleaned_df[columns_to_scale])

    # verification - plot distributions per variable
    # print(cleaned_df.head())
    fig = cleaned_df[columns_to_scale].hist(column=columns_to_scale, bins=100)
    [x.title.set_size(10) for x in fig.ravel()]
    plt.suptitle("Histogram distribution per variable", fontsize = 14)
    plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, wspace=0.6, hspace=0.6)
    plt.savefig('results/histograms.png')

    # write to csv
    cleaned_df.to_csv('cleaned_normalized.csv')

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

def weekday_dummies(data):

    for i in range(7):
        colname = 'weekdaydummy' + str(i)
        data[colname] = data['time'].apply(lambda x: x.weekday() == i)

    return data
