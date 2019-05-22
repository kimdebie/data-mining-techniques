import pandas as pd
import numpy as np

import load
import eda
import process
import features
import models
from sklearn.model_selection import KFold
import correlations

# global variables that define what tasks to perform
READ_RAW_DATA = False
PLOT = False
HYPERPARAM = True
LAMBDAMART = False
SAMPLING_METHOD = "downsample" # one of "downsample", "upsample", "none"


def main():

    # only read raw data if so required (cleaned files do not exist yet)
    if READ_RAW_DATA:

        # train set
        dataset = '../../data/training_set_VU_DM.csv'

        # test set (turn off relevance score in this case!)
        #dataset = '../../data/test_set_VU_DM.csv'

        # take the first 1000 lines of the dataset only - use this for testing
        # to make the code less slow! Comment it out for finalizing
        # dataset = '../data/testfile.csv'

        # loading in the right file
        data = load.loaddata(dataset)

        # # create competitor features
        data = features.create_competitor_features(data)

        # # create other features
        data = features.other_features(data)

        # # add relevance grades
        data = features.relevance_score(data)

        # remove outliers
        data = eda.remove_outliers(data)

        # # handling missing values
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

            # get correlations of the features
            correlations.show_correlations(sample_data)

        # divide data into train and test set (and save these)
        train_data, test_data = process.split_train_test(data)

        # downsample data to create class balance (and save)
        downsampled_train_data = process.downsample(train_data)

        # upsample data to create class balance (and save it)
        upsampled_train_data = process.upsample(train_data)


    # data is already loaded - only need to load it from file
    # test for the best set of hyperparameters
    if HYPERPARAM:

        # get the appropriate training set
        if SAMPLING_METHOD == "downsample":

            traindataset = '../data/downsampled_crossvalidation_set.csv'

        elif SAMPLING_METHOD == "upsample":

            traindataset = "../data/upsampled_crossvalidation_set.csv"

        elif SAMPLING_METHOD == "none":

            traindataset = "../data/full_crossvalidation_set.csv"

        # loading in the data
        train_data = load.loaddata(traindataset)

        # remove columns not in test dataset
        keep_cols = [col for col in train_data.columns if col not in ['booking_bool', 'click_bool']]

        # sample a smaller subset to make this all feasible
        train_data = train_data[keep_cols].sample(n=4000)
        print(train_data.columns)

        # Train lambdamart for different hyperparam values and evaluate on validation set
        trees = [5, 10, 50, 100, 150, 300, 400]
        lrs = [0.15, 0.10, 0.8, 0.05, 0.01]

        indices = []
        for i in range(np.array(train_data.shape[0])):
            items = [0, 1]
            indices.append(items)

        indices = np.array(indices)

        # K-fold cross validation for different parameter combinations
        for tree in trees:
            for lr in lrs:
                # indices = np.array(train_data.shape[0])
                kf = KFold(n_splits = 5)

                ndcgs = []
                for train_index, test_index in kf.split(indices):

                    train_index = train_index.tolist()
                    test_index = test_index.tolist()

                    # Split up data
                    X_train, X_validation = train_data.iloc[train_index], train_data.iloc[test_index]

                    # Run lambdamart on training data and evaluate on validation data
                    ndcg = models.lambdamart(X_train, X_validation, tree, lr, SAMPLING_METHOD)
                    print(ndcg)
                    ndcgs.append(ndcg)

                average_ndcg = np.mean(ndcgs)

                # Save NDCG
                file = '../results/hyperparams/crossvalidation_' + SAMPLING_METHOD + '.txt'
                with open(file, 'a') as f:
                    line = 'trees: ' + str(tree) + ', lr: ' + str(lr) + ', average_ndcg: ' + str(average_ndcg) + '\n'
                    print(line)
                    f.write(line)
                f.close()

        # run the full model
        if LAMBDAMART:

            # test data is always the same
            testdataset = '../data/testing_set_only.csv'

            # get the appropriate training set
            if SAMPLING_METHOD == "downsample":

                traindataset = '../data/downsampled_training_set_only.csv'

            elif SAMPLING_METHOD == "upsample":

                traindataset = "../data/upsampled_training_set_only.csv"

            elif SAMPLING_METHOD == "none":

                traindataset = "../data/full_training_set_only.csv"

            # loading in the data
            train_data = load.loaddata(traindataset)

            # loading in final test set
            test_data = load.loaddata(testdataset)

            # hyperparameters
            trees = 2
            lrs = 0.10

            # train lambdamart and evaluate on test set
            ndcg = models.lambdamart(train_data, test_data, trees, lrs, SAMPLING_METHOD)
            print(ndcg)


if __name__ == '__main__':

    main()
