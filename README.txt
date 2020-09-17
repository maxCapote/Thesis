file structure
    the directories noted as 'raw' contain the .csv files for the corresponding dataset as I received them
    the directories noted as 'clean' contain processed .csv files that are used for training and testing
        the structure of a 'clean' directory is as follows
            dataset_clean
                binary
                    training and testing sets
                multi
                    training and testing sets
code
    all functions have corresponding comments above them, except in stub
    there are files for the following
        preprocessing
        each model
        a stub for demonstration purposes
    each source code file is configured to work within the directory laid out in the manner I deliver this project in
        relative paths
    executing either model without editing the file would perform binary classification against CICIDS2017
        a new model would be created and trained first, so this may be a time-consuming process
        when needing to test multiclass classification or test against BoT-IoT, there are values I would change manually in the code
        note that the models have different configurations depending on the dataset and type of classification
            documented in the paper
for a quick demonstration, run the stub program
    this will load the two .dat files, which are trained xgboost models
        one for binary classification and the other for multiclass classification
        both trained on CICIDS2017
    some cherry-picked examples are used from CICIDS2017
        I just took two benign instances and the first two unique attack instances
        I thought this would be sufficient enough for a proof-of-concept
