"""
Last amended: 06th May, 2020


Objectives:
        
       i)  Learning to work with Words and Sequences
	   ii) Word-level one-hot encoding
	   iii) Character-level one-hot encoding
	   iv) Using Keras for word-level one-hot encoding
	   v) Word-level one-hot encoding with hashing

"""

# 1.0 Call libraries
%reset -f
import numpy as np
import pandas as pd
import os, shutil

#Word-level one-hot encoding
samples = ['The cat sat on the mat.', 'The dog ate my homework.']
#Initial data: one entry per sample (in this example, a sample is a sentence,
# but it could be an entire document)

#Tokenizes the samples via the splitmethod. 
#In real life, you’d also strip punctuation and special characters from the samples.

token_index = {}
for sample in samples:
    for word in sample.split():
        if word not in token_index:
            token_index[word] = len(token_index) + 1
			#Assigns a unique index to each unique word. 
			#Note that you don’t attribute index 0 to anything.
			
max_length = 12
#Vectorizes the samples. You’ll only consider the first max_length words in each sample.
results = np.zeros(shape=(len(samples),
                          max_length,
                          max(token_index.values()) + 1))
#This is where you store the results.

for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = token_index.get(word)
        results[i, j, index] = 1.
		
		
#Character-level one-hot encoding
import string
samples = ['The cat sat on the mat.', 'The dog ate my homework.']
characters = string.printable
token_index = dict(zip(range(1, len(characters) + 1), characters))
max_length = 50
results = np.zeros((len(samples), max_length, max(token_index.keys()) + 1))
for i, sample in enumerate(samples):
    for j, character in enumerate(sample):
        index = token_index.get(character)
        results[i, j, index] = 1.
		
#Using Keras for word-level one-hot encoding
from keras.preprocessing.text import Tokenizer

samples = ['The cat sat on the mat.', 'The dog ate my homework.']
tokenizer = Tokenizer(num_words= 50)
#Creates a tokenizer, configured to only take into account the 1,000 most common words

tokenizer.fit_on_texts(samples)
#Builds the word index

sequences = tokenizer.texts_to_sequences(samples)
#Turns strings into lists of integer indices

one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')
#You could also directly get the one-hot binary representations. 
# Vectorization modes other than one-hot encoding are supported by this tokenizer.
word_index = tokenizer.word_index
#How you can recover the word index that was computed

print('Found %s unique tokens.' % len(word_index))

#Word-level one-hot encoding with hashing trick (toy example)
dimensionality = 1000
''' Stores the words as vectors of size 1,000. If you have close
to 1,000 words (or more), you’ll see many hash collisions,
which will decrease the accuracy of this encoding method.'''
max_length = 10
results = np.zeros((len(samples), max_length, dimensionality))
for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = abs(hash(word)) % dimensionality
		''' Hashes the word into a
            random integer index between 0 and 1,000'''
        results[i, j, index] = 1.
		
