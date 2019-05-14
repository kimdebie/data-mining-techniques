import load
import features
import xgboost as xgb
from sklearn.metrics import accuracy_score

def lambda_mart(train, test):

    '''XGBoost model for ranking known as LambdaMART'''

    X_train = train.drop('relevance', axis=1)
    y_train = train['relevance']
    X_test = test.drop('relevance', axis=1)
    y_test = test['relevance'].as_matrix()

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # hyperparameters
    param = {'max_depth':5, 'eta':0.01, 'objective':'rank:ndcg' }
    num_round = 20

    # specify model
    model = xgb.train(param, dtrain, num_round)

    # predict
    preds = model.predict(dtest)
    return preds, y_test


if __name__ == '__main__':

    traindataset = '../data/downsampled_training_set.csv'
    train_data = load.loaddata(traindataset)
    train_data = train_data.drop('date_time', axis=1)
    train_data = features.relevance_score(train_data)

    testdataset = '../data/test_subset.csv'
    test_data = load.loaddata(testdataset)
    test_data = test_data.drop('date_time', axis=1)
    test_data = features.relevance_score(test_data)

    y_pred, y_test = lambda_mart(train_data, test_data)
    print(len(y_test))
    print(len(y_pred))
    #print('Accuracy: {0:.4f}'.format(accuracy_score(y_test, y_pred)))
