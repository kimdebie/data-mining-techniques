import pandas as pd

def benchmark(file):

    '''Implements a benchmark metric for predicting mood: mood is the same as the previous day'''

    data = pd.read_csv(file)

    data = data[["id", "time", "mood"]]

    # predicted mood is mood of previous day
    data["predicted_mood"] = data["mood"].shift(1)
    print(data.head())

    # we can only do this for observations with 1 day in between
    data["timegap"] = data["time"].diff() == 24
    data = data[data["time"].diff() == 24]


    print(data.head())




benchmark('cleaned_normalized.csv')
