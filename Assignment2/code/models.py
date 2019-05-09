import load
import process
import features
import pandas as pd
from  lamdamart import LambdaMART
import numpy as np


def lambdamart(train, test, nr_trees, lr): 

    # Transfer train-data to format for LambdaMart
    train_np, _ = load.lambdamartformat(train)
    test_np, idx_to_doc = load.lambdamartformat(test)

    # Train LambdaMart model
    model = LambdaMART(training_data=train_np, number_of_trees=nr_trees, learning_rate=lr)
    model.fit()

    # Save model for later use
    name = 'lambdamart_' + str(nr_trees) + '_' + str(lr)
    model.save(name)

	# Predict for test data
    predicted_scores = model.predict(test_np[:,1:])
    order = np.argsort(predicted_scores)    

	# Map ordered relevance scores to document ids
    ordered_docs = [idx_to_doc[idx] for idx in order]
    
    file = 'results/lambdamart_' + str(nr_trees) + '_' + str(lr) + '.txt'
    with open(file, 'w+') as f:
    	f.write('srch_id, prop_id \n')

    	# Get ranking for every query in test set
    	queries = test['srch_id'].unique().tolist()
    	for q in queries:
    		# Get documents for q
    		documents = test.loc[test['srch_id'] == q]['prop_id'].to_list()

    		# Get ranking based on relevance scores
    		ranking = [x for x in ordered_docs if x in documents]

    		# Append to documents (maybe only first 10??)
    		for doc in ranking:
    			line = str(q) + ', ' + str(doc) + '\n'
    			f.write(line)
    f.close()

