import preprocess
import analyze
import pivot
import pandas as pd
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from SVM import SVM_model, balanced_classes
from linearregr import linearregr
from benchmark import benchmark

filename = 'dataset_mood_smartphone.csv'
filename_clean = 'cleaned_normalized.csv'


def main():

    ############## PRE PROCESS DATA (only once) #############################
    data = preprocess.load(filename)
    clean_data = preprocess.clean(data)

    ############## READ CLEANED DATA ########################################
    data = pd.read_csv(filename_clean)
    data = data.drop(columns=["Unnamed: 0"])
    print(data.head())

    ############## EXTRACT FEATURES #########################################

    unobtrusive = data

    ## Create dataset including obtrusive features ##

    # removing all redundant columns / keeping those that we want features for
    cols_to_keep = ["id", "time", "mood", "sun", \
        "rain", "max_temp", "total_appuse", "activity", "circumplex.arousal", \
        "circumplex.valence", "weekdaydummy0", "weekdaydummy1", "weekdaydummy2", \
        "weekdaydummy3", "weekdaydummy4", "weekdaydummy5", "weekdaydummy6"]

    data = data[cols_to_keep]

    # creating lagged variables for the following columns (with defined durations)
    columns_to_lag = ["mood", "circumplex.arousal", "circumplex.valence", "total_appuse", "max_temp"]
    lags = [4, 3, 3, 3, 3]

    for i, col in enumerate(columns_to_lag):
        data = pivot.create_lagged_vars(data, col, lags=lags[i])

    # many rows are unusable so we drop them
    data = data.dropna()

    data.to_csv("with_features.csv")

    ## Creating unobtrusive-only dataset ##

    # removing all redundant columns / keeping those that we want features for
    un_cols_to_keep = ["id", "time", "mood", "sun", \
        "rain", "max_temp", "total_appuse", "activity", "weekdaydummy0", "weekdaydummy1", \
        "weekdaydummy2", "weekdaydummy3", "weekdaydummy4", "weekdaydummy5", "weekdaydummy6"]

    unobtrusive = unobtrusive[un_cols_to_keep]

    # creating lagged variables for the following columns (with defined durations)
    un_columns_to_lag = ["total_appuse", "max_temp"]
    lags = [4, 3]

    for i, col in enumerate(un_columns_to_lag):
        unobtrusive = pivot.create_lagged_vars(unobtrusive, col, lags=lags[i])

    # many rows are unusable so we drop them
    unobtrusive = unobtrusive.dropna()

    unobtrusive.to_csv("unobtrusive_with_features.csv")


    ## Correlations

    features = pd.read_csv('with_features.csv',index_col=0)
    correlations = calculate_pvalues(features)
    correlations.to_csv('correlations.csv')

    correlations = correlations.astype(float)
    correlations = correlations.drop(['time'], axis=1)
    correlations = correlations.drop(['time'], axis=0)
    sns.heatmap(correlations, vmin=0, vmax=1, center=0.5)
    plt.show()

def calculate_pvalues(df):
    df = df.dropna()._get_numeric_data()
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            p_value = round(pearsonr(df[r], df[c])[1], 4)
            pvalues[r][c] = p_value

            # for reading purposes, significant results are marked with an asterix
            # if p_value <= 0.05:
            #     pvalues[r][c] = str(p_value) + '*'
            #
            # else:
            #     pvalues[r][c] = p_value
    return pvalues
if __name__ == '__main__':

    main()

    data = pd.read_csv('with_features.csv',index_col=0)

    # create class labels for SVM
    n_classes = 4
    data = balanced_classes(data, data['mood'].as_matrix(), n_classes)

    # split data in training and test set
    msk = np.random.rand(len(data)) < 0.8
    train_data = data[msk]
    test_data = data[~msk]

    print(train_data.shape, test_data.shape)

    X_train = train_data.drop(['mood', 'label', 'id', 'time'], axis=1)
    X_test = test_data.drop(['mood', 'label', 'id', 'time'], axis=1)
    Y_train_svm = train_data['label']
    Y_train = train_data['mood']
    Y_test_svm = test_data['label']
    Y_test = test_data['mood']

    # perform experiments with different models
    mse, acc, correct_class_svm = SVM_model(X_train, X_test, Y_train_svm, Y_test_svm)
    print("SVM Accuracy: {}, MSE: {}".format(acc, mse))

    mse2, acc2, correct_class_regr = linearregr(X_train, X_test, Y_train, Y_test)
    print("Lin. Regression Accuracy: {}, MSE: {}".format(acc2, mse2))

    mse3, acc3, correct_class_bench = benchmark(test_data)
    print("Benchmark Accuracy: {}, MSE: {}".format(acc3,mse3))
