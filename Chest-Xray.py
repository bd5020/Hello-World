

#total  train samples for Normal : 	1341
#total  train samples for PNEUMONIA : 3875 
#total  validation samples for Normal : 	8
#total  validation samples for PNEUMONIA : 8 
#total  test samples for Normal : 234
#total  test samples for PNEUMONIA : 390 

%reset -f

# 1.0 Data manipulation library
#     Install in 'tf' environment
#     conda install -c anaconda pandas
import pandas as pd
import numpy as np

import os

# 1.1 Call libraries for image processing

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense

from tensorflow.keras import backend as K

# 1.4 Save CNN model configuration
from tensorflow.keras.models import model_from_json

# 1.6 For ROC plotting
import matplotlib.pyplot as plt

# conda install scikit-learn
from sklearn import metrics
import time
#from skimage import exposure           # Not used

img_width, img_height = 200, 200

# 2.2 Data folder containing all training images
train_data_dir = "D:\\Training\\Data\\Chest-Xray\\chest-xray-pneumonia\\chest_xray\\train"

nb_train_samples = 5216

# 2.4 Data folder containing all validation images

validation_data_dir = "D:\\Training\\Data\\Chest-Xray\\chest-xray-pneumonia\\chest_xray\\val"

nb_validation_samples = 16

# 2.6 Batch size to train at one go:
batch_size = 48             

# 2.7 How many epochs of training?
epochs = 1

# 2.8 No of test samples
test_generator_samples = 624

# 2.9 For test data, what should be batch size
test_batch_size = 32

input_shape = (img_width, img_height, 3)

model = Sequential()

model.add(Conv2D(
	             filters=48,                        
                 kernel_size=(3, 3),              
	             strides = (1,1),                 
	             input_shape=input_shape,          
	             use_bias=True,                    
	             padding='same',                  
	             name="Ist_conv_layer",
				 activation='relu'
	             )
         )
		 
model.add(Conv2D(
	             filters=64,                       	                                             
	             kernel_size=(3, 3),               
	             strides = (1,1),            
	             use_bias=True,                     
	             padding='same',                   
	             name="IInd_conv_layer",
				 activation='relu'              ##sigmoid 
	             )
         )

model.add(MaxPool2D())

model.summary()

Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
Ist_conv_layer (Conv2D)      (None, 240, 240, 48)      1344      
_________________________________________________________________
IInd_conv_layer (Conv2D)     (None, 240, 240, 64)      27712     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 120, 120, 64)      0         
=================================================================
Total params: 29,056
Trainable params: 29,056
Non-trainable params: 0

model.add(Conv2D(
	             filters=80,        ##42                
                 kernel_size=(3, 3),              
	             strides = (1,1),                 
	             use_bias=True,                    
	             padding='same',                  
	             name="3rd_conv_layer",
				 activation='relu'
	             )
         )
		 
model.add(Conv2D(
	             filters=80,                       	                                             
	             kernel_size=(3, 3),               
	             strides = (1,1),            
	             use_bias=True,                     
	             padding='same',                   
	             name="4th_conv_layer",
				 activation='relu'              ##sigmoid 
	             )
         )

model.add(MaxPool2D())

model.summary()

Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
Ist_conv_layer (Conv2D)      (None, 240, 240, 48)      1344      
_________________________________________________________________
IInd_conv_layer (Conv2D)     (None, 240, 240, 64)      27712     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 120, 120, 64)      0         
_________________________________________________________________
3rd_conv_layer (Conv2D)      (None, 120, 120, 80)      46160     
_________________________________________________________________
4th_conv_layer (Conv2D)      (None, 120, 120, 80)      57680     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 60, 60, 80)        0         
=================================================================
Total params: 132,896
Trainable params: 132,896
Non-trainable params: 0


# Adding a classifier on top of the model

model.add(Flatten(name = "FlattenedLayer"))
model.add(Dense(105))
model.add(Activation('relu'))
model.add(Dense(78))
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))
# 4.23
model.summary()

Model: "sequential_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
Ist_conv_layer (Conv2D)      (None, 200, 200, 48)      1344      
_________________________________________________________________
IInd_conv_layer (Conv2D)     (None, 200, 200, 64)      27712     
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 100, 100, 64)      0         
_________________________________________________________________
3rd_conv_layer (Conv2D)      (None, 100, 100, 80)      46160     
_________________________________________________________________
4th_conv_layer (Conv2D)      (None, 100, 100, 80)      57680     
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 50, 50, 80)        0         
_________________________________________________________________
FlattenedLayer (Flatten)     (None, 200000)            0         
_________________________________________________________________
dense_5 (Dense)              (None, 105)               21000105  
_________________________________________________________________
activation_6 (Activation)    (None, 105)               0         
_________________________________________________________________
dense_6 (Dense)              (None, 78)                8268      
_________________________________________________________________
activation_7 (Activation)    (None, 78)                0         
_________________________________________________________________
dense_7 (Dense)              (None, 1)                 79        
_________________________________________________________________
activation_8 (Activation)    (None, 1)                 0         
=================================================================
Total params: 21,141,348
Trainable params: 21,141,348
Non-trainable params: 0
_________________________________________________________________
_________________________________________________________________

# 4.25 Compile model
model.compile(
              loss='binary_crossentropy',  # Metrics to be adopted by convergence-routine
              optimizer='rmsprop',         # Strategy for convergence?
              metrics=['accuracy'])        # Metrics, I am interested in

model.summary()




#%%                            D. Create Data generators

def preprocess(img):
   return img


tr_dtgen = ImageDataGenerator(
                              rescale=1. / 255,      # Normalize colour intensities in 0-1 range
                              shear_range=0.2,       # Shear varies from 0-0.2
                              zoom_range=0.2,
                              horizontal_flip=True,
                              preprocessing_function=preprocess
                              )


train_generator = tr_dtgen.flow_from_directory(
                                               train_data_dir,       # Data folder of cats & dogs
                                               target_size=(img_width, img_height),  # Resize images
                                               batch_size=batch_size,  # Return images in batches
                                               class_mode='binary'   # Output labels will be 1D binary labels
                                                                     # [1,0,0,1]
                                                                     # If 'categorical' output labels will be
                                                                     # 2D OneHotEncoded: [[1,0],[0,1],[0,1],[1,0]]
                                                                     # If 'binary' use 'sigmoid' at output
                                                                     # If 'categorical' use softmax at output

                                                ) ##Found 5216 images belonging to 2 classes.

val_dtgen = ImageDataGenerator(rescale=1. / 255)

# 5.4.2 validation data

validation_generator = val_dtgen.flow_from_directory(
                                                     validation_data_dir,
                                                     target_size=(img_width, img_height),   # Resize images
                                                     batch_size=batch_size,    # batch size to augment at a time
                                                     class_mode='binary'  # Return 1D array of class labels
                                                     ) ##Found 16 images belonging to 2 classes.


## 6. Model fitting

# 6.1 Manual process of fitting. Get infinite images
#     Can experiment with infinite images. We will
#     generate upto 3200 images
#     https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit

start = time.time()   # 6 minutes
for e in range(epochs):
    print('Epoch', e)
    batches = 0
    for x_batch, y_batch in train_generator:
        model.fit(x_batch, y_batch)
        batches += 1
        print ("Epoch: {0} , Batches: {1}".format(e,batches))
        if batches > 200:    # 200 * 16 = 3200 images
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break

end = time.time()
(end - start)/60

Train on 48 samples
48/48 [==============================] - 11s 224ms/sample - loss: 0.4083 - accuracy: 0.8333
Epoch: 0 , Batches: 200
Train on 48 samples
48/48 [==============================] - 10s 218ms/sample - loss: 0.2753 - accuracy: 0.9375
Epoch: 0 , Batches: 201
Out[180]: 42.104426670074


# 7.0 Model evaluation

# 7.1 Using generator
#     https://www.tensorflow.org/api_docs/python/tf/keras/Model#evaluate
result = model.evaluate(validation_generator,
                                  verbose = 1,
                                  steps = 4        # How many batches
                                  )

WARNING:tensorflow:sample_weight modes were coerced from
  ...
    to  
  ['...']
1/4 [======>.......................] - ETA: 3s - loss: 0.4357 - accuracy: 0.7500WARNING:tensorflow:Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches (in this case, 4 batches). You may need to use the repeat() function when building your dataset.
1/4 [======>.......................] - ETA: 6s - loss: 0.2723 - accuracy: 0.7500

# 7.1.1

result
Out[182]: [0.10892064869403839, 0.75]												 