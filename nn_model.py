# load data
from pandas import read_csv

# processing data
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler

# neural net functionality
from tensorflow import keras

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
        the feature training set and feature testing set to be used
    return value(s)
        two numpy arrays
    function
        scale the features for the purpose of normalizing the data
"""
def scale(x_train, x_test):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test

"""
    parameters
        the target label training set and target label testing set to be used
    return value(s)
        two numpy arrays
    function
        binarize the categorical labels before use with neural network
"""
def binarize(y_train, y_test):
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.transform(y_test)
    return y_train, y_test

"""
    parameters
        none
    return value(s)
        a sequential keras model (the untrained neural network)
    function
        build and a compile the neural network model
"""
def create_model():
    model = keras.models.Sequential([
        keras.layers.Dense(256, activation='relu', input_dim=70),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    return model

"""
    parameters
        four numpy arrays, each one being a necessary training or testing set of data
    return value(s)
        a list of the performance metrics of the created model against the testing data
    function
        create, train, and evaluate a neural network model
"""
def train_eval_model(x_train, y_train, x_test, y_test):
    model = create_model()
    model.fit(x_train, y_train, epochs=10, batch_size=512)
    return model.evaluate(x_test, y_test)

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
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]        
        metrics.append(train_eval_model(x_train, y_train, x_test, y_test))
        print('Fold Completed\n')
    mean_metrics = np.mean(metrics, axis=0)
    print('Mean Values Across Metrics')
    print('\tAccuracy: {}'.format(mean_metrics[1]))
    print('\tPrecision: {}'.format(mean_metrics[2]))
    print('\tRecall: {}'.format(mean_metrics[3]))

"""
    parameters
        none
    return value(s)
        none
    function
        main function of the program
"""
def main():
    # force tensorflow to use CPU over GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    x_train, y_train, x_test, y_test = load_data(SECONDARY_DIR)
    x_train, x_test = scale(x_train, x_test)
    y_train, y_test = binarize(y_train, y_test)

    # uncomment to perform cross-validation
    #cross_validate(x_train, y_train)
    
    # uncomment to perform official training and evaluation
    train_eval_model(x_train, y_train, x_test, y_test)

if __name__ == '__main__':
    main()
