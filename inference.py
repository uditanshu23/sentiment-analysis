# -- coding: utf-8 --
'''
Created on 24-12-2022 13:29
Project : sentiment-analysis
@author : Uditanshu Satpathy
@emails : uditanshusatpathy23@gmail.com
'''

import os

from utils.processor import *
from utils.inference import *

def main():
    path = os.path.join(os.getcwd(), 'data', 'airline_sentiment_analysis.csv')
    processor = Processor()
    train_df, test_df = processor.data_processor(path)
    inference = Inference()
    inference.inference(test_df)

if __name__ == '__main__':
    main()