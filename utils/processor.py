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
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

class Processor:

    def __init__(self):
        self.STOPWORDS = set(stopwords.words('english'))
        english_punctuations = string.punctuation
        self.punctuations_list = english_punctuations

    def read_data(self, path):
        #read the csv file 

        data = pd.read_csv(path)
        return data
    
    def cleaning_stopwords(self, text):
        #remove the stop words from the text

        return " ".join([word for word in str(text).split() if word not in self.STOPWORDS])

    def cleaning_repeating_char(text):
        #remove the repitative characters from the text

        return re.sub(r'(.)1+', r'1', text)
    
    def cleaning_punctuations(self, text):
        #remove the punctuations like ! and @ and # from the text

        return text.translate(str.maketrans('', '', self.punctuations_list))

    def data_processor(self, path):
        df = self.read_data(path)
        data = df[['text', 'airline_sentiment']]
        train_df, test_df = train_test_split(data, test_size=0.2, random_state=42, shuffle=True)

        train_df['text']=train_df['text'].str.lower()
        train_df['text'] = train_df['text'].apply(lambda text: self.cleaning_stopwords(text))
        train_df['text'] = train_df['text'].apply(lambda x: self.cleaning_repeating_char(x))
        train_df['text']= train_df['text'].apply(lambda x: self.cleaning_punctuations(x))

        test_df['text']=test_df['text'].str.lower()
        test_df['text'] = test_df['text'].apply(lambda text: self.cleaning_stopwords(text))
        test_df['text'] = test_df['text'].apply(lambda x: self.cleaning_repeating_char(x))
        test_df['text']= test_df['text'].apply(lambda x: self.cleaning_punctuations(x))

        return train_df, test_df