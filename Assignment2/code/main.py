import load
import eda
import process
import features
import pandas as pd

# global variables that define what tasks to perform
READ_RAW_DATA = True
PERFORM_EDA = True
REMOVE_OUTLIERS = True


def main():

    # only read raw data if so required (cleaned files do not exist yet)
    if READ_RAW_DATA:

        dataset = '../data/training_set_VU_DM.csv'

        # take the first 1000 lines of the dataset only - use this for testing
        # to make the code less slow! Comment it out for finalizing
        dataset = '../data/testfile.csv'

        # loading in the right file
        data = load.loaddata(dataset)

        if PERFORM_EDA:

            # handling missing values
            data = eda.missing_values(data)

            if REMOVE_OUTLIERS:

                # remove outliers
                data = eda.remove_outliers(data)

            else:
                data = load.loaddata('../data/outliers_removed.csv')


            # create competitor features
            data = features.create_competitor_features(data)

            # take a sample of the data to make plotting feasible
            sample_data = data.sample(n=500000)

            # plot distributions
            eda.plot_distributions(sample_data)

            # plot correlations between sets of variables
            eda.plot_correlations(sample_data)

            # plot impact of price of competitor on booking
            eda.plot_competitor_price_impact(sample_data)

        # divide data into train and test set
        train_data, test_data = process.split_train_test(data)

        # downsample data to create class balance
        train_data = process.downsample(train_data)

    # data is already loaded - only need to load it from file
    else:

        traindataset = '../data/downsampled_training_set.csv'
        train_data = load.loaddata(traindataset)

        testdataset = '../data/test_subset.csv'
        test_data = load.loaddata(testdataset)




if __name__ == '__main__':

    main()
