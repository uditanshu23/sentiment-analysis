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
    
        def __init__(self, checkpoint_path="models/cp-{epoch:04d}.ckpt", batch_size=32, epochs=1):
            self.checkpoint_path = checkpoint_path
            self.batch_size = batch_size
            self.epochs = epochs
        
        def create_model(self, vocab_size, embedding_vector_length=32, loss='binary_crossentropy', optimizer='adam'):
            model = Sequential() 
            model.add(Embedding(vocab_size, embedding_vector_length, input_length=200) )
            model.add(SpatialDropout1D(0.25))
            model.add(LSTM(50, dropout=0.5, recurrent_dropout=0.5))
            model.add(Dropout(0.2))
            model.add(Dense(1, activation='sigmoid')) 
            model.compile(optimizer, loss, metrics=['accuracy'])  
            print(model.summary()) 
            return model
        
        def train(self, train_df):

            sentiment_lab = train_df.airline_sentiment.factorize()

            text = train_df.text.values
            tokenizer = Tokenizer(num_words=5000)
            tokenizer.fit_on_texts(text)
            vocab_size = len(tokenizer.word_index) + 1
            print(vocab_size)
            encoded_docs = tokenizer.texts_to_sequences(text)
            padded_sequence = pad_sequences(encoded_docs, maxlen=200)

            cp_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=self.checkpoint_path, 
                verbose=1, 
                save_weights_only=True,
                save_freq=2*self.batch_size)
            
            model = self.create_model(vocab_size)
            model.save_weights(self.checkpoint_path.format(epoch=0))

            prediction = model.fit(padded_sequence, sentiment_lab[0], epochs=self.epochs, validation_split=0.2, batch_size=32, callbacks=[cp_callback])

            plt.plot(prediction.history['accuracy'], label='accuracy')
            plt.plot(prediction.history['val_accuracy'], label='val_accuracy')
            plt.savefig('./results/accuracy plot.jpg')
            return prediction

        def train_save(self, train_df):
            pred = self.train(train_df)
            
            plt.plot(pred.history['loss'], label='loss')
            plt.plot(pred.history['val_loss'], label='val_loss')
            plt.savefig("./results/Loss plot.jpg")
