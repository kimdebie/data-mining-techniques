import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def plot_hist(data):

    cols = cols_to_keep = ["mood", "sun", \
        "rain", "max_temp", "total_appuse", "activity", "circumplex.arousal", \
        "circumplex.valence"]

    fig = data.hist(column=cols, bins=100)
    [x.title.set_size(10) for x in fig.ravel()]
    plt.suptitle("Histogram distribution per variable", fontsize = 14)
    plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, wspace=0.6, hspace=0.6)
    plt.savefig('results/histograms.png')


data = pd.read_csv("with_features.csv")
plot_hist(data)
