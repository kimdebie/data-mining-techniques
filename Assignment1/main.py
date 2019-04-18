import preprocess
import analyze
import pivot
import pandas as pd
from scipy.stats import pearsonr

filename = 'dataset_mood_smartphone.csv'
filename_clean = 'cleaned_normalized.csv'

def calculate_pvalues(df):
    df = df.dropna()._get_numeric_data()
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            p_value = round(pearsonr(df[r], df[c])[1], 4)
            if p_value <= 0.05:
                pvalues[r][c] = str(p_value) + '*'
            else:
                pvalues[r][c] = p_value
    return pvalues

def main():

############## PRE PROCESS DATA (only once) #############################
    # data = preprocess.load(filename)
    # clean_data = preprocess.clean(data)

############## READ CLEANED DATA ########################################
    data = pd.read_csv(filename_clean)
    data = data.drop(columns=["Unnamed: 0"])
    print(data.head())

############## EXTRACT FEATURES #########################################
    correlations = calculate_pvalues(data)
    correlations.to_csv('correlations.csv')

    unobtrusive = data

    # removing all redundant columns / keeping those that we want features for
    cols_to_keep = ["id", "time", "mood", "weekday", "sun", \
        "rain", "max_temp", "total_appuse", "activity", "circumplex.arousal", \
        "circumplex.valence"]

    data = data[cols_to_keep]

    # creating lagged variables for the following columns (with defined durations)
    columns_to_lag = ["mood", "circumplex.arousal", "circumplex.valence", "total_appuse", "max_temp"]
    lags = [4, 3, 3, 3, 3]

    for i, col in enumerate(columns_to_lag):
        data = pivot.create_lagged_vars(data, col, lags=lags[i])

    # many rows are unusable so we drop them
    data = data.dropna()

    data.to_csv("with_features.csv")



    # Creating unobtrusive-only dataset

    # removing all redundant columns / keeping those that we want features for
    un_cols_to_keep = ["id", "time", "mood", "weekday", "sun", \
        "rain", "max_temp", "total_appuse", "activity"]

    unobtrusive = unobtrusive[un_cols_to_keep]

    # creating lagged variables for the following columns (with defined durations)
    un_columns_to_lag = ["total_appuse", "max_temp"]
    lags = [4, 3]

    for i, col in enumerate(un_columns_to_lag):
        unobtrusive = pivot.create_lagged_vars(unobtrusive, col, lags=lags[i])

    # many rows are unusable so we drop them
    unobtrusive = unobtrusive.dropna()

    unobtrusive.to_csv("unobtrusive_with_features.csv")



if __name__ == '__main__':
    main()
