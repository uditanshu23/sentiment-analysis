# -- coding: utf-8 --
'''
Created on 24-12-2022 11:00
Project : sentiment-analysis
@author : Uditanshu Satpathy
@emails : uditanshusatpathy23@gmail.com
'''

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
from sklearn.model_selection import train_test_split


import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense, Dropout, SpatialDropout1D
from tensorflow.keras.layers import Embedding

class Train:
    
        def __init__(self, checkpoint_path="sentiment_analysis/cp-{epoch:04d}.ckpt", batch_size=32):
            self.checkpoint_path = checkpoint_path
            self.batch_size = batch_size