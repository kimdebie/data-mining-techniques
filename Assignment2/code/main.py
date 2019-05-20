import pandas as pd
import numpy as np

import load
import eda
import process
import features
import models
import correlations

# global variables that define what tasks to perform
READ_RAW_DATA = False
PLOT = False
SAMPLING_METHOD = "downsample" # one of "downsample", "upsample", "none"


def main():

    # only read raw data if so required (cleaned files do not exist yet)
    if READ_RAW_DATA:

        dataset = '../data/training_set_VU_DM.csv'

        # take the first 1000 lines of the dataset only - use this for testing
        # to make the code less slow! Comment it out for finalizing
        #dataset = '../data/testfile.csv'

        # loading in the right file
        data = load.loaddata(dataset)

        # create competitor features
        data = features.create_competitor_features(data)

        # create other features
        data = features.other_features(data)

        # add relevance grades
        data = features.relevance_score(data)

        # create competitor features
        data = features.create_competitor_features(data)

        # create other features
        data = features.other_features(data)

        # add relevance grades
        data = features.relevance_score(data)

        # remove outliers
        data = eda.remove_outliers(data)

        # handling missing values
        data = eda.missing_values(data)

        if PLOT:

            # take a sample of the data to make plotting feasible
            sample_data = data.sample(n=500000)

            # plot distributions
            eda.plot_distributions(sample_data)

            # plot correlations between sets of variables
            eda.plot_correlations(sample_data)

            # plot impact of price of competitor on booking
            eda.plot_competitor_price_impact(sample_data)

        # divide data into train and test set (and save these)
        train_data, test_data = process.split_train_test(data)

        # downsample train data to create class balance (and save it)
        downsampled_train_data = process.downsample(train_data)

        # upsample data to create class balance (and save it)
        upsampled_train_data = process.upsample(train_data)


    # data is already loaded - only need to load it from file
    # when training models, start from here!
    else:

        # test data is always the same
        testdataset = '../data/testing_set.csv'

        # get the appropriate training set
        if SAMPLING_METHOD == "downsample":

            traindataset = '../data/downsampled_training_set.csv'

        elif SAMPLING_METHOD == "upsample":

            traindataset = "../data/upsampled_training_set.csv"

        elif SAMPLING_METHOD == "none":

            traindataset = "../data/full_training_set.csv"

        # loading in the data
        train_data = load.loaddata(traindataset)
        test_data = load.loaddata(testdataset)

        # get correlations of the features
        correlations.show_correlations(train_data)

        # Train lambdamart and evaluate on test set
        models.lambdamart(train_data, test_data, 2, 0.10)

if __name__ == '__main__':

    main()
