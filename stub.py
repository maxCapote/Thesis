from pandas import read_csv
from pandas import concat
from joblib import load
import os
import time

PRIMARY_DIR = os.path.join(os.curdir, 'CICIDS2017_CLEAN')
BIN_DIR = os.path.join(PRIMARY_DIR, 'BINARY')
MUL_DIR = os.path.join(PRIMARY_DIR, 'MULTI')
BIN_MODEL = os.path.join(os.curdir, 'xgboost_binary.dat')
MUL_MODEL = os.path.join(os.curdir, 'xgboost_multi.dat')

def load_data(directory):
    x = read_csv(os.path.join(directory, 'x_test.csv'))
    y = read_csv(os.path.join(directory, 'y_test.csv'))

    demo_x = concat([
        x.loc[[0]],
        x.loc[[1]],
        x.loc[[9]],
        x.loc[[414]]
    ])

    demo_y = concat([
        y.loc[[0]],
        y.loc[[1]],
        y.loc[[9]],
        y.loc[[414]]
    ])

    return demo_x, demo_y

def demo(directory, model_name):
    x, y = load_data(directory)
    model = load(model_name)

    y_pred = model.predict(x)
    print('[+] predictions for test instances:')
    print(y_pred, '\n')
    time.sleep(4)

    print('[+] actual values of test instances:')
    print(y.values.ravel(), '\n')
    time.sleep(4)

def main():
    demo(BIN_DIR, BIN_MODEL)
    demo(MUL_DIR, MUL_MODEL)

if __name__ == '__main__':
    main()
