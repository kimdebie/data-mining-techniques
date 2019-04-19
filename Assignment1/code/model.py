import pandas as pd
import numpy as np

# split the dataset by 'chunkID', return a dict of id to rows
"""
Get chunks of data. Chunk_ix denotes the id column.
"""
def to_chunks(values, chunk_ix=0):
    chunks = dict()
    # get the unique chunk ids
    chunk_ids = np.unique(values[:, chunk_ix])
    # group rows by chunk id
    for chunk_id in chunk_ids:
        selection = values[:, chunk_ix] == chunk_id
        chunks[chunk_id] = values[selection, :]

    return chunk_ids, chunks

"""
Split dataset. First 40 days are used for training (so time < 40 * 24).
row_in_chunk indicates the time column that we want to split on
"""
def split_train_test(chunks, row_in_chunk_ix=1):
    train, test = list(), list()

    # first 40 days for training
    cut_point = 40 * 24

    # split dataset
    for k,rows in chunks.items():
        train_rows = rows[rows[:,row_in_chunk_ix] <= cut_point, :]
        test_rows = rows[rows[:,row_in_chunk_ix] > cut_point, :]

        # remove chunks with insufficient data
        if len(train_rows) == 0 or len(test_rows) == 0:
            print("Drop chunk {}".format(k))
            continue

        train.append(train_rows)
        test.append(test_rows)

    return train, test

def average_mood(ids, data):
    for id in ids:
        rows = data.loc[data['id'] == id]
        mean_rows = rows.mean()
        print(mean_rows)

# read data
data = pd.read_csv('cleaned_normalized.csv', header = 0)
data = data.drop(columns=["Unnamed: 0"])

# group data by chunks
values = data.values
ids, chunks = to_chunks(values)
print('Total Chunks: %d' % len(chunks))

# get training and test data
train, test = split_train_test(chunks)
print(train_df)
average_mood(ids,train)

print('Train Rows: %s' % str(train_rows.shape))
print('Test Rows: %s' % str(test_rows.shape))
