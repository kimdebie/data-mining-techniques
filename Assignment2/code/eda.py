import pandas as pd
import matplotlib.pyplot as plt

def missing_values(df):

    '''Plot and handle missing values.'''

    # Plot the columns with missing values
    df.isna().sum().sort_values(ascending=False)[df.isna().sum().sort_values() > 0].plot.bar()
    plt.title("Count of missing values per variable")
    plt.ylabel("Number of values missing")
    plt.show()

    # remove variables containing too many NAs? Something else?

    return df


def plot_distributions(df):

    '''Plot distributions of relevant variables.'''

    # plot all columns, excluding booleans, ids, competitor columns, flags and position
    cols_to_plot = [col for col in df.columns if 'bool' not in col and 'id' not in col \
                and 'comp' not in col and 'flag' not in col and 'position' not in col]

    df[cols_to_plot].hist(bins=30)
    plt.show()

    # plot competitor columns
    comp_cols = [col for col in df.columns if '_rate_percent_diff_mag' in col]
    df[comp_cols].hist(bins=30)
    plt.show()


def create_competitor_features(df):

    '''Create features for the competitor columns.'''


    # instead of absolute difference with competitor, specify this to difference
    # with magnitude: is Expedia cheaper or more expensive?
    for i in range(1, 9):

        ratecol = "comp" + str(i) + "_rate"
        pricecol = "comp" + str(i) + "_rate_percent_diff"
        magncol = "comp" + str(i) + "_rate_percent_diff_mag"

        df[magncol] = df[ratecol] * df[pricecol]

        print(i)

    return df
