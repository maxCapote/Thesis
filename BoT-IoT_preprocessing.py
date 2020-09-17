# load and prepare data
from pandas import read_csv
from pandas import concat

# process data
from pandas import to_numeric
from pandas import get_dummies
from sklearn.model_selection import train_test_split

# utility
import os

"""
    parameters
        a dataframe object and a list of target features for cleaning
    return value(s)
        a dataframe object
    function
        clean the non-numeric values in given features for the given dataframe object
"""
def clean_ports(df, columns):
    for column in columns:
        for i in df.index:
            if type(df.at[i, column]) == str and df.at[i, column].startswith('0x'):
                # convert hex value in a string to decimal value
                df.at[i, column] = int(df.at[i, column], 16)
        # all other objects (str) are converted to int too
        df[column] = to_numeric(df[column])
    return df

"""
    parameters
        the directory containing the appropriate files
    return value(s)
        a dataframe object
    functions
        load data from files into dataframe objects
        clean the non-numeric values in 'sport' and 'dport' for each object
        compile the objects into a single dataframe
"""
def load_data(directory):
    # load data
    files = [read_csv(os.path.join(directory, file)) for file in os.listdir(directory)]
    # convert hex values in 'sport' and 'dport' to int
    for i in range(0, len(files)):
        files[i] = clean_ports(files[i], ['sport', 'dport'])
    # compile data into one dataframe
    return concat(files)

"""
    parameters
        a dataframe object
    return value(s)
        a dataframe object
    function
        one-hot encode categorical features in the dataset
"""
def encode_features(dataset):
    encoded_dataset = dataset
    for feature in dataset:
        if(dataset[feature].dtype == object):
            one_hot = get_dummies(dataset[feature])
            encoded_dataset = concat([encoded_dataset, one_hot], axis=1)
            encoded_dataset = encoded_dataset.drop(columns=feature)
    return encoded_dataset

"""
    parameters
        a dataframe object
    return value(s)
        a dataframe object
    function
        update the provided target labels to reflect binary classification
"""
def update_labels_binary(labels):
    attack_labels = {
        'DDoS':'Attack',
        'DoS':'Attack',
        'Reconnaissance':'Attack',
        'Theft':'Attack'
    }
    return labels.map(attack_labels).fillna('Normal')

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
        'DoS':'DoS/DDoS',
        'Reconnaissance':'Reconnaissance',
        'Theft':'Theft'
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
    data_raw = os.path.join(os.curdir, 'BoT-IoT_RAW')
    data_clean = os.path.join(os.curdir, 'BoT-IoT_CLEAN')

    dataset = load_data(data_raw)
    
    x, y = concat([dataset.iloc[:,:-2], dataset.iloc[:,-1]], axis=1), dataset.iloc[:,-2]
    x = encode_features(x)  
    y_bin = update_labels_binary(y)
    y_multi = update_labels_multi(y)
    
    divide_save(x, y_bin, os.path.join(data_clean, 'BINARY'))
    divide_save(x, y_multi, os.path.join(data_clean, 'MULTI'))

if __name__ == '__main__':
    main()
