'''
Functions for preprocessing the data
'''

import pandas as pd

def load(filename):

    data = pd.read_csv(filename)
    print(data.head())

    return data


def clean(data):

    return clean_data


def create_features_ML(clean_data):

    raise NotImplementedError

    return featurized_data


def create_features_temporal(clean_data):

    raise NotImplementedError

    return featurized_data
