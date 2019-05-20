import pandas as pd
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
import numpy as np

def loaddata(filename):

    '''Loading in data.'''

    df = pd.read_csv(filename)

    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)

    # print(df.dtypes)

    return df

def lambdamartformat(data, test = False):
	idx_to_doc = {}
	new_data = []

	features = list(data.columns.values)
	features.sort()

	if 'relevance' in features:	
		features.remove('relevance')
	features.remove('srch_id')
	features.remove('prop_id')
	features.remove('date_time')

	# We should do something with time!!
	# features.remove('')

	idx = 0
	for index, row in data.iterrows():
		new_arr = []
		if test == False:
			new_arr.append(row['relevance'])
		else:
			new_arr.append(0)
		new_arr.append(row['srch_id'])
		for feature in row[features].values:
			new_arr.append(feature)
		new_data.append(new_arr)
		idx_to_doc[idx] = row['prop_id']
		idx += 1

	return np.array(new_data), idx_to_doc
