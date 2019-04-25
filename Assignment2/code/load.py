import pandas as pd
import matplotlib.pyplot as plt

def loaddata(filename):

    '''Loading in data.'''

    df = pd.read_csv(filename)

    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)

    return df
