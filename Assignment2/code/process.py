import pandas as pd
from sklearn.model_selection import train_test_split

def split_train_test(df, train_size=0.8, test_size=0.2):

    '''Randomly split data in training and test set.'''

    train_set, test_set = train_test_split(df, test_size=test_size)

    # write test set to file
    test_set.to_csv('../data/test_subset.csv')

    return train_set, test_set

def downsample(df):

    '''Downsample such that classes are balanced.'''

    # separate booked, clicked and neither
    booked = df[df['booking_bool'] == 1]
    clicked = df[(df.click_bool == 1) & (df.booking_bool == 0)]
    neither = df[(df.click_bool == 0) & (df.booking_bool == 0)]

    # determine which subset has the lowest number of observations
    minimum_observations = min([len(booked.index), len(clicked.index), len(neither.index)])

    # sample the minimum number of observations from each category
    booked_sampled = booked.sample(n=minimum_observations)
    clicked_sampled = clicked.sample(n=minimum_observations)
    neither_sampled = neither.sample(n=minimum_observations)

    # combine into one df again
    downsampled_df = pd.concat([booked_sampled, clicked_sampled, neither_sampled])

    # write to csv
    downsampled_df.to_csv('../data/downsampled_training_set.csv')

    return downsampled_df
