# load and prepare data
from pandas import read_csv
from pandas import concat

# process data
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split

# utility
import os
import numpy as np

"""
    parameters
        the directory containing the appropriate files
    return value(s)
        a dataframe object
    function
        load data from files into dataframe objects and compile the objects into a single dataframe
"""
def load_data(directory):
    # load data
    files = [read_csv(os.path.join(directory, file)) for file in os.listdir(directory)]
    # compile data into one dataframe
    return concat(files)

"""
    parameters
        a dataframe object
    return value(s)
        a dataframe object
    function
        remove infinity and missing values from the dataset
"""
def clean_infs_missing(dataset):
    # replace inf values with nan values
    dataset = dataset.replace([np.inf, np.NINF], np.nan)
    # drop missing values
    return dataset.dropna()

"""
    parameters
        a dataframe object
    return value(s)
        a dataframe object
    function
        remove features from the dataset that contain no variance
"""
def drop_zero_variance(features):
    selector = VarianceThreshold()
    selector.fit(features)
    return features.iloc[:,selector.variances_ > 0]

"""
    parameters
        a dataframe object
    return value(s)
        a dataframe object
    function
        update the provided target labels to reflect binary classification
"""
def update_labels_binary(labels):
    return labels.map({'BENIGN':'Normal'}).fillna('Attack')

"""
    parameters
        a dataframe object
    return value(s)
        a dataframe object
    function
        update the provided target labels to reflect multiclass classification
"""
def update_labels_multi(labels):
    attack_labels = {
        'DDoS':'DoS/DDoS',
        'DoS Hulk':'DoS/DDoS',
        'DoS GoldenEye':'DoS/DDoS',
        'DoS slowloris':'DoS/DDoS',
        'DoS Slowhttptest':'DoS/DDoS',
        'Heartbleed':'DoS/DDoS',
        'PortScan':'Port Scan',
        'FTP-Patator':'Brute-Force',
        'SSH-Patator':'Brute-Force',
        'Web Attack � Brute Force':'Web Attack',
        'Web Attack � XSS':'Web Attack',
        'Web Attack � Sql Injection':'Web Attack',
        'Bot':'Botnet ARES',
        'Infiltration':'Infiltration'
    }
    return labels.map(attack_labels).fillna('Normal')

"""
    parameters
        two dataframe objects and a directory name
    return value(s)
        none
    function
        divide the dataframe objects into training and testing sets and save the sets to file
"""
def divide_save(x, y, directory):
    # divide into training and testing sets (80/20)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
    # save training and testing sets
    x_train.to_csv(os.path.join(directory, 'x_train.csv'), index=False)
    x_test.to_csv(os.path.join(directory, 'x_test.csv'), index=False)
    y_train.to_csv(os.path.join(directory, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(directory, 'y_test.csv'), index=False)

"""
    parameters
        none
    return value(s)
        none
    function
        main function of the program
"""
def main():
    data_raw = os.path.join(os.curdir, 'CICIDS2017_RAW')
    data_clean = os.path.join(os.curdir, 'CICIDS2017_CLEAN')

    dataset = load_data(data_raw)
    dataset = clean_infs_missing(dataset)

    x, y = dataset.iloc[:,:-1], dataset.iloc[:,-1]
    x = drop_zero_variance(x)
    y_bin = update_labels_binary(y)
    y_multi = update_labels_multi(y)

    divide_save(x, y_bin, os.path.join(data_clean, 'BINARY'))
    divide_save(x, y_multi, os.path.join(data_clean, 'MULTI'))

if __name__ == '__main__':
    main()
