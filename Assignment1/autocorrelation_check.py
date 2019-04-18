import pandas as pd
from pandas.plotting import autocorrelation_plot
from pandas.plotting import lag_plot
import matplotlib.pyplot as plt
import pivot

def autocorrelation(file):

    '''Checking for autocorrelation within mood variable''''

    df = pd.read_csv(file)

    df = df[["id", "time", "mood"]]

    for id in df["id"].unique():

        series = df[df["id"] == id].mood


        autocorrelation_plot(series)
        plt.title("Autocorrelation plot for user " + id)
        plt.show()


        lag_plot(series)
        plt.xlabel("Mood at current timepoint")
        plt.ylabel("Mood at next timepoint")
        plt.title("Lag plot for user " + id)
        plt.show()

def corr_with_lag(file, col, lags=5):

    '''Checking whether lags of other variables correlate with current mood'''

    data = pd.read_csv(file)

    data = data[["id", "time", "mood", col]]
    print(data.head())

    data_lag = pivot.create_lagged_vars(data, col, lags)

    for lag in range(lags):

        colname = col + "_lag" + str(lag+1)
        data_lag.plot(x='mood', y=colname, style='o')
        plt.show()

    print(data_lag.head())
    corrs = data_lag.corr()
    print(corrs["mood"])

    return data_lag

#autocorrelation('cleaned_normalized.csv')
corr_with_lag('cleaned_normalized.csv', 'rain')
