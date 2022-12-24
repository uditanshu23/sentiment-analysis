# -- coding: utf-8 --
'''
Created on 22-12-2022 21:11
Project : utils
@author : Uditanshu Satpathy
@emails : uditanshusatpathy23@gmail.com
'''

import re
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.model_selection import train_test_split

class Processor:
    def __init__(self):
        pass

    def read_data(self, path):
        data = pd.read_csv(path)
        return data

    def assign_flag(self, data):
        flags = []
        for ind in data.index:
            fla = (1 if data['airline_sentiment'][ind]== 'positive' else 0)
            flags.append(fla)
        data['flag'] = flags
        return data