"""
Last amended: 07th May, 2020


Objectives:
        
       i)  Learning to work with Words and Sequences
	   ii) Processing the labels of the raw IMDB data
	   iii) Using Embedding Layer
	   

"""

# 1.0 Call libraries
%reset -f
import numpy as np
import pandas as pd
import os, shutil

#Instantiating an Embedding layer (Just an Example of instantiating)#
from keras.layers import Embedding
embedding_layer = Embedding(1000, 64)
'''
The Embedding layer takes at least two arguments: 
the number of possible tokens (here, 1,000: 1 + maximum word index)
and the dimensionality of the embeddings (here, 64).
'''
#Word index  -----> Embedding layer ----> Corresponding word vector

#Loading the IMDB data for use with an Embedding layer#
from keras.datasets import imdb
from keras import preprocessing

max_features = 10000 #Number of words to consider as features
maxlen = 20 #Cuts off the text after this number of words (among
            #the max_features most common words)

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
#Loads the data as lists of integers

x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
#Turns the lists of integers into a 2D integer tensor of shape (samples, maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

#Using an Embedding layer and classifier on the IMDB data#
from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()

model.add(Embedding(10000, 8, input_length=maxlen))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid')) 
#corresponding to 160 inputs from flatten layer
# and one threshold > 160+1 = 161 params  

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()
 
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding_2 (Embedding)      (None, 20, 8)             80000
_________________________________________________________________
flatten_1 (Flatten)          (None, 160)               0
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 161
=================================================================
Total params: 80,161
Trainable params: 80,161
Non-trainable params: 0
_________________________________________________________________
'''

history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_split=0.2)

'''
You get to a validation accuracy of ~76%, which is pretty good considering that you’re
only looking at the first 20 words in every review. But note that merely flattening the
embedded sequences and training a single Dense layer on top leads to a model that
treats each word in the input sequence separately, without considering inter-word
relationships and sentence structure (for example, this model would likely treat both
“this movie is a bomb” and “this movie is the bomb” as being negative reviews).
'''

'''
It’s much better to add recurrent layers or 1D convolutional layers on top of 
the embedded sequences to learn features that take into account each sequence as a whole.
'''


