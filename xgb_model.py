# load data
from pandas import read_csv

# xgboost functionality
from xgboost import XGBClassifier

# evaluation metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# cross-validation for fine-tuning
from sklearn.model_selection import KFold

# utility
import os
import numpy as np

# the primary directory denotes which dataset to use
#   'CICIDS2017_CLEAN' and 'BoT-IoT_CLEAN' may be specified to use a particular dataset
PRIMARY_DIR = os.path.join(os.curdir, 'CICIDS2017_CLEAN')
# the secondary directory denotes the use of the binary or multiclass data
#   'BINARY' is specified when using binary sets
#   'MULTI' is specified when using multiclass sets
SECONDARY_DIR = os.path.join(PRIMARY_DIR, 'BINARY')

"""
    parameters
        the directory containing the appropriate files
    return value(s)
        four dataframe objects
    function
        load data from files into dataframe objects
"""
def load_data(directory):
    filenames = ['x_train.csv', 'y_train.csv', 'x_test.csv', 'y_test.csv']
    files = [read_csv(os.path.join(directory, file)) for file in filenames]
    return files[0], files[1], files[2], files[3]

"""
    parameters
        none
    return value(s)
        an xgboost model (the untrained ensemble model)
    function
        build the xgboost model
"""
def create_model():
    return XGBClassifier(
        n_estimators=125,
        learning_rate=0.3,
        max_depth=10,
        gamma=0.5,
        subsample=1,
        colsample_bytree=1,
        objective='binary:logistic',
        nthread=4,
        seed=42
    )

"""
    parameters
        four numpy arrays, each one being a necessary training or testing set of data
    return value(s)
        a list of the performance metrics of the created model against the testing data
    function
        create, train, and evaluate an xgboost model
    note
        pos_label='Normal' is used for binary classification
        average='micro' is used for multiclass classification, replacing pos_label='Normal'
"""
def train_eval_model(x_train, y_train, x_test, y_test):
    model = create_model()
    model.fit(x_train, y_train.values.ravel())
    y_pred = model.predict(x_test)
    return [accuracy_score(y_test.values.ravel(), y_pred),
            precision_score(y_test.values.ravel(), y_pred, pos_label='Normal'),
            recall_score(y_test.values.ravel(), y_pred, pos_label='Normal')]

"""
    parameters
        a list containing the metrics to display
    return value(s)
        none
    function
        display the performance metrics passed
"""
def display_metrics(metrics):
    print('\tAccuracy: {}'.format(metrics[0]))
    print('\tPrecision: {}'.format(metrics[1]))
    print('\tRecall: {}'.format(metrics[2]))

"""
    parameters
        a feature set and target label set
    return value(s)
        none
    function
        conduct 4-fold cross-validation and display the mean values across metrics
"""
def cross_validate(x, y):
    metrics = []
    for train_index, test_index in KFold(n_splits=4, shuffle=True, random_state=42).split(x, y):
        print('Beginning Fold...')
        x_train, x_test = x.loc[train_index], x.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]
        fold_results = train_eval_model(x_train, y_train, x_test, y_test)
        display_metrics(fold_results)
        metrics.append(fold_results)
        print('Fold Completed\n')
    mean_metrics = np.mean(metrics, axis=0)
    print('Mean Values Across Metrics')
    display_metrics(mean_metrics)

"""
    parameters
        none
    return value(s)
        none
    function
        main function of the program
"""
def main():
    x_train, y_train, x_test, y_test = load_data(SECONDARY_DIR)

    # uncomment to perform cross-validation
    #cross_validate(x_train, y_train)

    # uncomment to perform official training and evaluation
    print('Evaluation Metrics')
    display_metrics(train_eval_model(x_train, y_train, x_test, y_test))

if __name__ == '__main__':
    main()
