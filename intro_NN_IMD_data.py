"""

Objectives:
        i) Experiments in Keras
       ii) Learning to work with Neural Network 

"""

# 1.0 Call libraries
%reset -f
import pandas as pd
import os

#Loading the IMDB dataset
from keras.datasets import imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
num_words=10000)

#The argument num_words=10000 means you’ll only keep the top 10,000 most frequently
# occurring words in the training data. Rare words will be discarded

"""
The variables train_data and test_data are lists of reviews; each review is a list of
word indices (encoding a sequence of words). train_labels and test_labels are
lists of 0s and 1s, where 0 stands for negative and 1 stands for positive
"""
train_data[0]      #[1, 14, 22, 16, ... 178, 32]
train_labels[0]    #1

max([max(sequence) for sequence in train_data])   #9999
#no word index will exceed 10,000

#how you can quickly decode one of these reviews back to English words:
word_index = imdb.get_word_index()
reverse_word_index = dict(
[(value, key) for (key, value) in word_index.items()])
decoded_review = ' '.join(
[reverse_word_index.get(i - 3, '?') for i in train_data[0]])

#word_index is a dictionary mapping words (key) to an integer(value) index
'''
Decodes the review. Note that the indices
are offset by 3 because 0, 1, and 2 are
reserved indices for “padding,” “start of
sequence,” and “unknown.”
'''
# Encoding the integer sequences into a binary matrix

import numpy as np
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
	#Creates an all-zero matrix of shape (len(sequences), dimension)
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
		#Sets specific indices of results[i] to 1s
    return results
	
x_train = vectorize_sequences(train_data) #Vectorized training data
x_test = vectorize_sequences(test_data) #Vectorized test data

x_train[0]   #array([ 0., 1., 1., ..., 0., 0., 0.])

#You should also vectorize your labels
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

'''
There are two key architecture decisions to be made about such a stack of Dense layers:
1. How many layers to use
2. How many hidden units to choose for each layer
'''

#The model definition
from keras import models
from keras import layers
model = models.Sequential()
model.add(layers.Dense(32, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

#choose a loss function and an optimizer

'''
Crossentropy is a quantity from the field of Information Theory
that measures the distance between probability distributions or, in this
case, between the ground-truth distribution and your predictions
'''
#Compiling the model
model.compile(optimizer='rmsprop',
loss='binary_crossentropy',
metrics=['accuracy'])

#Using custom losses and metrics
from keras import losses
from keras import metrics
from keras import optimizers
model.compile(optimizer=optimizers.RMSprop(lr=0.001),
loss=losses.binary_crossentropy,
metrics=[metrics.binary_accuracy])

#Setting aside a validation set
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

#Training your model
model.compile(optimizer='rmsprop',
loss='binary_crossentropy',
metrics=['acc'])
history = model.fit(partial_x_train,
partial_y_train,
epochs=20,
batch_size=512,
validation_data=(x_val, y_val))

'''
The call to model.fit() returns a History object. This object has a member
history, which is a dictionary containing data about everything that happened
during training.
'''
history_dict = history.history
history_dict.keys()
#[u'acc', u'loss', u'val_acc', u'val_loss']
#The dictionary contains four entries: one per metric that was being monitored during
#training and during validation

#Plotting the training and validation loss
import matplotlib.pyplot as plt
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
#epochs = range(1, len(acc) + 1)
epochs = range(1, 20 + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']
#epochs = range(1, len(acc) + 1)
epochs = range(1, 20 + 1)
plt.plot(epochs, acc_values, 'bo', label='Training acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.title('Training and validation acc')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()



