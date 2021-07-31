# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 05:18:23 2020

@author: lenovo
"""

import numpy as np
from string import punctuation

with open('reviews.txt','r') as f:
    reviews = f.read()

with open('labels.txt', 'r') as f:
    labels = f.read()


reviews = reviews.lower()
all_texts = ''.join([c for c in reviews if c not in punctuation])
reviews_split = all_texts.split("\n")
print("number of reviews", len(reviews_split))
labels = labels.lower()
labels_split = ''.join([c for c in labels if c not in punctuation])
all_labels = labels_split.split("\n")


from collections import Counter
all_text2 = ' '.join(reviews_split)
# create a list of words
words = all_text2.split()# Count all the words using Counter Method
count_words = Counter(words)
total_words = len(words)
sorted_words = count_words.most_common(total_words)

vocab_to_int = {w:i+1 for i, (w,c) in enumerate(sorted_words)}

reviews_int = []
for review in reviews_split:
    r = [vocab_to_int[w] for w in review.split()]
    reviews_int.append(r)


#Encoding labels as positive : 1, negative : 0
encoded_labels = [1 if label =='positive' else 0 for label in all_labels]
encoded_labels = np.array(encoded_labels)

import pandas as pd
import matplotlib.pyplot as plt
reviews_len = [len(x) for x in reviews_int]
pd.Series(reviews_len).hist()
plt.show()
pd.Series(reviews_len).describe()
 #Get rid of outlier
reviews_int = [ reviews_int[i] for i, l in enumerate(reviews_len) if l>0 ]
encoded_labels = [ encoded_labels[i] for i, l in enumerate(reviews_len) if l> 0 ]
maxlength = 250 #max length of  reviews
seq_length  = maxlength
def pad_features(reviews_int, maxlength):
    ''' Return features of review_ints, where each review is padded with 0's or truncated to the input seq_length.
    '''
    features = np.zeros((len(reviews_int), seq_length), dtype = int)
    
    for i, review in enumerate(reviews_int):
        review_len = len(review)
        
        if review_len <= seq_length:
            zeroes = list(np.zeros(seq_length-review_len))
            new = zeroes+review        
        elif review_len > seq_length:
            new = review[0:seq_length]
        
        features[i,:] = np.array(new)
    
    return features

reviews_int = pad_features(reviews_int, maxlength)
np.asarray(reviews_int)
print("shape", np.shape(reviews_int))
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(reviews_int, encoded_labels, test_size=0.33, random_state=42)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)
X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
print("X train shape",np.shape(X_train))
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
print('Build model...')

model = Sequential()
model.add(LSTM(256, dropout=0.5, recurrent_dropout=0.5))
model.add(Dense(1, activation='softmax'))

X_train = X_train.reshape(np.shape(X_train)[0], 250, 1)
X_test = X_test.reshape(np.shape(X_test)[0], 250, 1)

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
batch_size = 100
print('Train...')
model.fit(X_train,y_train,
          batch_size=batch_size,
          epochs=15,
          )
score, acc = model.evaluate(X_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)





