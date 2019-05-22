import load
import process
import features
import pandas as pd
import lambdamart as fn
import numpy as np


def lambdamart(train, test, nr_trees, lr, datatype, write=False, hyperparams = True):

    # Transfer train-data to format for LambdaMart
    train_np, _ = load.lambdamartformat(train)
    test_np, idx_to_doc = load.lambdamartformat(test, True)

    # Train LambdaMart model
    model = fn.LambdaMART(training_data=train_np, number_of_trees=nr_trees, learning_rate=lr)
    model.fit()

    # Save model for later use
    name = 'results/models/lambdamart_' + str(nr_trees) + '_' + str(lr)
    model.save(name)

	# Predict for test data
    # predicted_scores = model.predict(test_np[:,1:])
    avg_ndcg, predicted_scores = model.validate(test_np, 10)
    #predicted_scores = model.predict(test_np[:,1:])

    order = np.argsort(predicted_scores)

	# Map ordered relevance scores to document ids
    ordered_docs = [idx_to_doc[idx] for idx in order]

    # Save ranking
    file = 'results/lambdamart_' + str(nr_trees) + '_' + str(lr) + '.txt'
    with open(file, 'w+') as f:
        f.write('srch_id, prop_id \n')

        ndcgs = []

        # Get ranking for every query in test set
        queries = test['srch_id'].unique().tolist()

        for q in queries:
            # Get documents for q
            documents = test.loc[test['srch_id'] == q]['prop_id'].to_list()
            relevance = test.loc[test['srch_id'] == q]['relevance'].tolist()

            k = len(documents)

            # Get ranking based on relevance scores from lambdamart
            ranking = [x for x in ordered_docs if x in documents]

            # Calculate ndcg
            dcg = fn.dcg_k(relevance, k)
            idcg = fn.ideal_dcg_k(relevance, k)

            print(relevance)

            if idcg == 0:
                ndcg = 0.0
            else:
                ndcg = dcg / idcg

            ndcgs.append(ndcg)

        average_ndcg = np.mean(ndcgs)
        #     if write == True:
        #         for doc in ranking:
        #             line = str(q) + ',' + str(doc) + '\n'
        #             f.write(line)

        # f.close()

    return average_ndcg

