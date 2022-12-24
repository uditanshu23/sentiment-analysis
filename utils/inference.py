# -- coding: utf-8 --
'''
Created on 24-12-2022 13:10
Project : sentiment-analysis
@author : Uditanshu Satpathy
@emails : uditanshusatpathy23@gmail.com
'''

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense, Dropout, SpatialDropout1D
from tensorflow.keras.layers import Embedding

class Inference:

    def __init__(self, checkpoint_path = "sentiment_analysis/cp-{epoch:04d}.ckpt"):
        self.checkpoint_path = checkpoint_path

    def create_model(vocab_size, embedding_vector_length=32, loss='binary_crossentropy', optimizer='adam'):
        model = Sequential() 
        model.add(Embedding(vocab_size, embedding_vector_length, input_length=200) )
        model.add(SpatialDropout1D(0.25))
        model.add(LSTM(50, dropout=0.5, recurrent_dropout=0.5))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid')) 
        model.compile(loss, optimizer, metrics=['accuracy'])  
        print(model.summary()) 
        return model

    def predict_sentiment(tokenizer, senti_lab, model, text):
        
        tw = tokenizer.texts_to_sequences([text])
        tw = pad_sequences(tw,maxlen=200)
        prediction = int(model.predict(tw).round().item())
        print("Predicted label: ", senti_lab[1][prediction])

        return None

    def inference(self,train_df, test_df):

        senti_lab = train_df.airline_sentiment.factorize()

        text = test_df.text.values
        tokenizer = Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(text)
        encoded_docs = tokenizer.texts_to_sequences(text)

        model = Inference.create_model(vocab_size=len(tokenizer.word_index) + 1)

        checkpoint_dir = os.path.dirname(self.checkpoint_path)
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        model.load_weights(latest)

        self.predict_sentiment(tokenizer, senti_lab, model, text)

        return None

