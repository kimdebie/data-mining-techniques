import pandas as pd
import matplotlib.pyplot as plt

def loaddata(filename):

    '''Loading in data.'''

    df = pd.read_csv(filename)
    df = df.drop('Unnamed: 0', axis=1)

    print(df.head())
    print(df.dtypes)

    return df
