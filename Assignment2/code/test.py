import pandas as pd
import features


files = ['../data/downsampled_crossvalidation_set.csv', '../data/upsampled_crossvalidation_set.csv', '../data/full_crossvalidation_set.csv',
'../data/downsampled_training_set_only.csv', '../data/full_training_set_only.csv', '../data/upsampled_training_set_only.csv']

files = ['../data/testing_set_only.csv']

for file in files:

    df = pd.read_csv(file)

    df = features.relevance_score(df)
    # separate booked, clicked and neither
    booked = df[df['relevance'] == 0]
    clicked = df[df['relevance'] == 1]
    neither = df[df['relevance'] == 5]

    print(len(booked.index))
    print(len(clicked.index))
    print(len(neither.index))

    df.to_csv(file)
