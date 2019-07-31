name = "example_pkg"
import os
import sys
ABSPATH = os.path.abspath(os.path.realpath(os.path.dirname(__file__)))
sys.path.append(ABSPATH)

from auto_smart.model import Model
import numpy as np
import pandas as pd
import json
from os.path import join
from datetime import datetime


TYPE_MAP = {
    'time': str,
    'cat': str,
    'multi-cat': str,
    'num': np.float64
}

def read_info(datapath):
    with open(join(datapath, 'train', 'info.json'), 'r') as info_fp:
        info = json.load(info_fp)
    return info

def read_train(datapath, info):
    train_data = {}
    for table_name, columns in info['tables'].items():

        table_dtype = {key: TYPE_MAP[val] for key, val in columns.items()}

        if table_name == 'main':
            table_path = join(datapath, 'train', 'main_train.data')
        else:
            table_path = join(datapath, 'train', f'{table_name}.data')

        date_list = [key for key, val in columns.items() if val == 'time']

        train_data[table_name] = pd.read_csv(
            table_path, sep='\t', dtype=table_dtype, parse_dates=date_list,
            date_parser=lambda millisecs: millisecs if np.isnan(
                float(millisecs)) else datetime.fromtimestamp(
                    float(millisecs)/1000))

    # get train label
    train_label = pd.read_csv(
        join(datapath, 'train', 'main_train.solution'))['label']
    return train_data, train_label


def read_test(datapath, info):
    # get test data
    main_columns = info['tables']['main']
    table_dtype = {key: TYPE_MAP[val] for key, val in main_columns.items()}

    table_path = join(datapath, 'test', 'main_test.data')

    date_list = [key for key, val in main_columns.items() if val == 'time']

    test_data = pd.read_csv(
        table_path, sep='\t', dtype=table_dtype, parse_dates=date_list,
        date_parser=lambda millisecs: millisecs if np.isnan(
            float(millisecs)) else datetime.fromtimestamp(
                float(millisecs) / 1000))
    return test_data

def train_and_predict(train_data,train_label,info,test_data):
    cmodel = Model(info)
    cmodel.fit(train_data, train_label)
    return cmodel.predict(test_data)

