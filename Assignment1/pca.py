import pandas as pd
from pandas.plotting import autocorrelation_plot
from pandas.plotting import lag_plot
import matplotlib.pyplot as plt

def autocorrelation(file):

    df = pd.read_csv(file)

    df = df[["id", "time", "mood"]]

    for id in df["id"].unique():

        series = df[df["id"] == id].mood


        autocorrelation_plot(series)
        plt.title("Autocorrelation plot for user " + id)
        plt.show()


        # lag_plot(series)
        # plt.xlabel("Mood at current timepoint")
        # plt.ylabel("Mood at next timepoint")
        # plt.title("Lag plot for user " + id)
        # plt.show()

autocorrelation('cleaned_normalized.csv')
