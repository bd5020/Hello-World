"""
Last amended: 07th May, 2020


Objectives:
        
       i)  Working with Words and Sequences
	   ii) Processing the labels of the raw IMDB data
	   iii) Using the pretrained layer
	   

"""

# 1.0 Call libraries
%reset -f
import numpy as np
import pandas as pd
import os, shutil

#Processing the labels of the raw IMDB data #

imdb_dir = 'E:/lalit/Teaching/New_courses/Course_for_FORE/BIG_Data_5/Lectures/Deep_Learning_for_Text_Seq/IMDB_raw_data/aclImdb'
train_dir = os.path.join(imdb_dir, 'train')

labels = []
texts = [] #both are lists

for label_type in ['neg', 'pos']:
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname), encoding="utf8")
            texts.append(f.read()) #Append on a new line
		    #list of sentences 
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)

##Tokenizing the text of the raw IMDB data ##

#Because pretrained word embeddings are meant to be particularly useful 
# on problems where little training data is available
#We are restricting the training data to the first 200 samples

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

maxlen = 100                 #Cuts off reviews after 100 words (of each review)
training_samples = 200       #Trains on 200 samples (i.e. case for a small data set)
validation_samples = 10000   #Validates on 10,000 samples
max_words = 10000            #Considers only the top
                             #10,000 words in the dataset

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts) #We are using raw text
sequences = tokenizer.texts_to_sequences(texts)
len(texts[0]) # 655  It treat each character as an element
len(sequences[0]) #104 It treat each word as an element

# i.e. texts > sequences conversion i.e. conversion to sequences of numbers

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=maxlen) #limiting to maxlen words
labels = np.asarray(labels)
print('Shape of data tensor:', data.shape) #(25000, 100)
print('Shape of label tensor:', labels.shape) #(25000,)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]


##Parsing the GloVe word-embeddings file##
#parse the unzipped file (a .txt file) to build an index that maps words #
#(as strings) to their vector representation (as number vectors)#


glove_dir = 'E:/lalit/Teaching/New_courses/Course_for_FORE/BIG_Data_5/Lectures/Deep_Learning_for_Text_Seq/Glove/glove.6B'
embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'), encoding="utf8")

for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
	
f.close()
print('Found %s word vectors.' % len(embeddings_index))

'''
Build an embedding matrix that you can load into an Embedding layer. 
It must be a matrix of shape (max_words, embedding_dim), where each entry i contains
the embedding_dim-dimensional vector for the word of index i in the reference word
index (built during tokenization).
'''

##Preparing the GloVe word-embeddings matrix##
embedding_dim = 100
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
		    #Words not found in the embedding index will be all zeros.

##Model definition##
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding_3 (Embedding)      (None, 100, 100)          1000000
_________________________________________________________________
flatten_2 (Flatten)          (None, 10000)             0
_________________________________________________________________
dense_2 (Dense)              (None, 32)                320032
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 33
=================================================================
Total params: 1,320,065
Trainable params: 1,320,065
Non-trainable params: 0
_________________________________________________________________

'''

'''
The Embedding layer has a single weight matrix: a 2D float matrix where each entry i is
the word vector meant to be associated with index i. Simple enough. Load the GloVe
matrix you prepared into the Embedding layer, the first layer in the model.
'''

##Loading pretrained word embeddings into the Embedding layer##
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False
#you’ll freeze the Embedding layer (set its trainable attribute to False),

##Training and evaluation##
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
			  
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_val, y_val))
					
model.save_weights('pre_trained_glove_model.h5')

##Plotting the results##
import matplotlib.pyplot as plt
%matplotlib qt5

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


##Training the same model without pretrained word embeddings##
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_val, y_val))

#let’s evaluate the model on the test data#
test_dir = os.path.join(imdb_dir, 'test')
labels = []
texts = []
for label_type in ['neg', 'pos']:
    dir_name = os.path.join(test_dir, label_type)
    for fname in sorted(os.listdir(dir_name)):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname))
            texts.append(f.read())
			f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)

sequences = tokenizer.texts_to_sequences(texts)
x_test = pad_sequences(sequences, maxlen=maxlen)
y_test = np.asarray(labels)

##Evaluating the model on the test set##
model.load_weights('pre_trained_glove_model.h5')
model.evaluate(x_test, y_test)

#You get an appalling test accuracy of 56%. 
#Working with just a handful of training samples is difficult!
