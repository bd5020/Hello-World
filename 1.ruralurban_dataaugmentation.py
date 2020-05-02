# Last amended: 30th April, 2020
# My folder:  /home/ashok/Documents/4. transfer_learning

#
# Ref https://www.kaggle.com/dansbecker/exercise-data-augmentation
# https://www.kaggle.com/learn/deep-learning
# How the images were collected:
# Images were collected using facility available in Moodle at
#    http://203.122.28.230/moodle/mod/url/view.php?id=1768
# Or see:
#    https://addons.mozilla.org/en-US/firefox/addon/google-images-downloader/

# ResNet architecture is here:  http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006

# Objective:
#           Predict Rural-urban images with data augmentation using ResNet50

"""
Steps:
     i) Call libraries
    ii) Define necessary constants
   iii) Create ResNet model and classifier
   iv)  Create nested model
    v)  Make certain layers of ResNet50 trainable
        THIS STEP REDUCES ACCURACY
   vi)  Compile model
  vii)  Train-Data Augmentation:
             i) Define operations to be done on images--Configuration 1
            ii) Define from where to read images from,
                batch-size,no of classes, class-model  --Configuration 2
	ix) Validation data augmentation
	x)  Start training--Model fitting
    xi) Check if resnet weights have changed or not

"""



#    cp /home/ashok/.keras/keras_tensorflow.json  /home/ashok/.keras/keras.json
#    cat /home/ashok/.keras/keras.json
#    source activate tensorflow

######################### Call libraries
# 1. Call libraries
%reset -f

# 1.1 Application (ResNet50) library
# https://keras.io/applications/#resnet50
from tensorflow.keras.applications import ResNet50

# 1.2 Keras models and layers
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,  GlobalAveragePooling2D

# 1.3 Image generator and preprocessing
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1.4 To save model-weights with least val_acc
#      Refer Page 250 Chollet

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import callbacks

# 1.4 Misc
import matplotlib.pyplot as plt
import time,os
import numpy as np

######################### Define necessary constants

# 2. Constants & Modeling
num_classes = 2
image_size = 224      # Restrict image-sizes to uniform: 224 X 224
                      # Decreasing increases speed but reduces accuracy

# 2.1 Image data folders
train_dir = "C:\\Users\\ashok\\Desktop\\chmod\\4.transfer_learning\\ruralurban\\train"
val_dir = "C:\\Users\\ashok\\Desktop\\chmod\\4.transfer_learning\\ruralurban\\val"
#train_dir= '/home/ashok/Images/ruralurban/train'
#val_dir = '/home/ashok/Images/ruralurban/val'


######################### Create ResNet model and classifier


# 2.2 Create ResNet59 model and import weights
resnet_base_model = ResNet50(
                          include_top=False,   # No last softmax layer
                          pooling='avg',       # GlobalAveragePooling2D to flatten last convolution layer
                                               # See: http://203.122.28.230/moodle/mod/resource/view.php?id=3740
                          weights='imagenet'
                          )

"""
Meaning of arguments
=====================
include_top = False
       whether to include the fully-connected layer at the top of the network.
pooling = 'avg'
       means that global average pooling will be
       applied to the output of the last convolutional layer,
       and thus the output of the model will be a 2D tensor.
       (2D includes batch_size also else it is just 1D)
weights:
       one of None (random initialization) or 'imagenet' (pre-training on ImageNet).

"""


# 2.3 Look at the model
resnet_base_model.summary()

# 2.4 Total number of layers are:

len(resnet_base_model.layers)      # 176
type(resnet_base_model.layers)     # A list of layers

# 2.5 Look at layer names again
for i,layer in enumerate(resnet_base_model.layers):
    print((i,layer.name))
    #print(layer.get_weights())    # Machine may hang


# 2.6 Initialise: Freeze all layers from training
for layer in resnet_base_model.layers:
    layer.trainable = False


# 2.7 Make layer 161 onwards available for training
#     Re-training some ResNet layers without sufficient data
#      may reduce speed as also accuracy

for layer in resnet_base_model.layers[160:]:
    layer.trainable = True


# 2.8 Quick Check. Number of True
#     should be 176 - 161 = 15 + 1 (includes 161 also)
for layer in resnet_base_model.layers:
    print(layer.trainable)


# 2.9 Record existing weights in any two ResNet layers
#     One having trainable as False and another having
#     trainable as True.
#     We will see if weights change after
#     modeling for trainable (True) layer and not
#     for trainable (False)
#     StackOverFlow: https://stackoverflow.com/questions/43715047/how-do-i-get-the-weights-of-a-layer-in-keras

# 2.9.1 Select two layers
resnet_base_model.layers[150]             # Conv2D layer
resnet_base_model.layers[150].trainable   # False

resnet_base_model.layers[171]             # Conv2D layer
resnet_base_model.layers[171].trainable   # True

# 2.9.2 Get weights of two layers
wt_150 = resnet_base_model.layers[150].get_weights()[0]
wt_171 = resnet_base_model.layers[171].get_weights()[0]
wt_150.shape                  # (1, 1, 512, 2048)
wt_171.shape                  # (1, 1, 512, 2048)

# 2.9.3 Are these sets of wts equal?
np.sum(wt_150 == wt_171)      # 0   No

########### Add classification layers #############

# 3 Start nested model building
my_new_model = Sequential()


# 3.1 Nest base model within it
my_new_model.add(resnet_base_model)


# 3.2
my_new_model.summary()

# 3.3 Last output softmax layer
my_new_model.add(Dense(
                       num_classes,
                       activation='softmax'
                       )
                )

# 3.4
my_new_model.summary()

# 3.5 my_new_model can also access nested layers
#     as here:
type(my_new_model.layers)      # list
len(my_new_model.layers)       # 2 List contains two objects
                               #  one, resnet50 and other dense layer
                               #   see model summary

# 3.5.1 Iterate over first object of list
for layer in my_new_model.layers[0].layers:
    print(layer.trainable)


# 4.0 Compile model
my_new_model.compile(
                     optimizer='sgd',      # One record at a time. It is fast
                     loss='categorical_crossentropy',  # Same as binary-crossentropy
                                                       # Used when labels are one-hot-encoded
                     metrics=['accuracy']
                     )



######################### Train-Data Augmentation

# 4.1 Image processing and image generation
#     train data Image generator object
data_generator_with_aug = ImageDataGenerator(
                                             preprocessing_function=preprocess_input,      # keras image preprocessing function
                                             horizontal_flip=True,
                                             width_shift_range = 0.2,
                                             height_shift_range = 0.2
                                            )

# 4.2 No fit() needed
#     Create image data generator interator for train data
train_generator = data_generator_with_aug.flow_from_directory(
                                                              train_dir,
                                                              target_size=(image_size, image_size),
                                                              batch_size=16 ,    # Increasing it increases
                                                                                 # processing time & may decrease accu
                                                              class_mode='categorical'  # labels are in OHE format
                                                              )


# 4.2.1 Looking at class labels
g = train_generator.__iter__()
g.next()[1]     # Index 0 is image. Labels are One-hot-encoded (OHE)

######################### Validation-Data Augmentation


# 4.3 validation data generator object
#     We will manipulate even Validation data also
#     Just to see if predictions are still made correctly
#     'data_generator_no_aug' is learner + ImageDataGenerator object
data_generator_no_aug = ImageDataGenerator(
                                           preprocessing_function=preprocess_input,
                                           rotation_range=90,
                                           horizontal_flip=True,
                                           vertical_flip=True,
                                           )


# 4.4 validation data image iterator
validation_generator = data_generator_no_aug.flow_from_directory(
                                                                 val_dir,
                                                                 target_size=(image_size, image_size),
                                                                 batch_size= 16,
                                                                 class_mode='categorical'
                                                                 )


######################### Model fitting

# 5. Prepare a list of callback functions
#    We will only have one. It will look at the val_loss
#    at the end of each epoch. Only if val_loss, for current
#    epoch is less than the previous, model-weights will be
#    saved, else not.
#    Refer page 250 Chollet

# 5.01
checkpoint_path = "C:\\Users\\ashok\\Desktop\\chmod\\4.transfer_learning\\"
#checkpoint_path = '/home/ashok/Documents/4.transfer_learning/'
filepath =  checkpoint_path + "my_model.h5"

#### &&&& #### This limit is important #### &&&& See para after 'fit'

# 5.02 Delete any existing file. Linux command
if os.path.exists(checkpoint_path):
    os.system('rm ' + filepath)

# 5.03 Delete any existing file. Windows command
if os.path.exists(checkpoint_path):
    os.system('del ' + filepath)

# 5.04 Check if "my_model.h5" exists?
os.path.exists(filepath)     # False

# 5.1 Create now callbacks list:
mycallbacks_list = [
                     ModelCheckpoint(filepath= filepath,
                                    monitor = 'val_loss',
                                    save_best_only = True
                                    )
                    ]

# 5.2 Model fitting. Takes 21 minutes for epochs = 5
#     When inner layers of ResNet50 are trained, accuracy
#     is less.

# 5.2.1   Break loop after some time at any point
#         If you are working in ipython, press ctrl+C to break
#         this loop. If in 'atom', just close 'atom'. And then
#         open atom again.

start = time.time()
history = my_new_model.fit(
                           train_generator,
                           steps_per_epoch=4,     #  Total number of batches of samples to
                                                  #   yield from generator before declaring one epoch
                                                  #     finished and starting the next epoch.
                                                  #   Increase it to train the model better
                           epochs=50,             # All steps when finished, constitute one epoch
						                         # Increase it to get better training
                           validation_data=validation_generator,
                           validation_steps=2,    #  Total number of steps (batches of samples)
                                                  #   to yield from validation_data generator per epoch
                                                  #    Increase it to get a better idea of validation accu
                           workers = 2,           # Maximum number of processes to spin up
                           callbacks=mycallbacks_list, # What callbacks to act upon after each epoch
                           verbose = 1            # Show progress

                           )

end = time.time()
print("Time taken: ", (end - start)/60, "minutes")


# 5.3 So how to begin now?
#     We will begin our work from saved weights
#     Execute all lines upto para 5.01 AND also
#     execute para 5.1. Continue next.

# 5.3.1 Load model weights saved by checkpointing
#       Have a look if model-weights have actually
#       been saved.

os.listdir(checkpoint_path)

# 5.3.2
my_new_model.load_weights(checkpoint_path + "my_model.h5")

# 5.3.3 Start fitting again. This time validation accuracy starts
#       higher up
start = time.time()
history = my_new_model.fit(
                           train_generator,
                           steps_per_epoch=4,     #  Total number of batches of samples to
                                                  #   yield from generator before declaring one epoch
                                                  #     finished and starting the next epoch.
                                                  #   Increase it to train the model better
                           epochs=10,             # All steps when finished, constitute one epoch
						                         # Increase it to get better training
                           validation_data=validation_generator,
                           validation_steps=2,    #  Total number of steps (batches of samples)
                                                  #   to yield from validation_data generator per epoch
                                                  #    Increase it to get a better idea of validation accu
                           workers = 2,           # Maximum number of processes to spin up
                           callbacks=mycallbacks_list, # What callbacks to act upon after each epoch
                           verbose = 1            # Show progress

                           )

end = time.time()
print("Time taken: ", (end - start)/60, "minutes")



# 5.3.4 So what has happened to recorded weights
#       after modeling
post_wt_150 = resnet_base_model.layers[150].get_weights()[0]
post_wt_171 = resnet_base_model.layers[171].get_weights()[0]
post_wt_150.shape                  # (1, 1, 512, 2048)
post_wt_171.shape                  # (1, 1, 512, 2048)

np.sum(post_wt_150 == wt_150)      # 1048576  (= 512 * 2048 ) No wt has changed
np.sum(post_wt_171 == wt_171)      # Most wts have changed. Few have not.


# 5.4 Plot training accuracy and validation accuracy
#     Next time use TensorBoard
plot_history()


# 5.4
#     How accuracy changes as epochs increase
#     We will use this function agai and again
#     in subsequent examples

def plot_history():
    val_acc = history.history['val_accuracy']
    tr_acc=history.history['accuracy']
    epochs = range(1, len(val_acc) +1)
    plt.plot(epochs,val_acc, 'b', label = "Validation accu")
    plt.plot(epochs, tr_acc, 'r', label = "Training accu")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.show()


###########################################################################



"""
 What does GlobalAveragePooling2D do?
 And what are its adavantages over normal FC layer?
  It takes overall average of every filter. So for a convolution layer
    with 32 filters, we will have a 1D layer with 32 neurons
     GlobalAveragePooling2D can, therefore, be used to flatten the last
      convolution layer. See also below at the end of this code.
      See: https://www.quora.com/What-is-global-average-pooling
	 : https://stats.stackexchange.com/a/308218

Effect of GlobalAveragePoolimg is that the last resnet50 layer is a flat layer
with 2048 neurons. See this link: http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006
The next layer is a dense layer with 2 neurons. Thus total number of weights are: 4098


_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
resnet50 (Model)             (None, 2048)              23587712
_________________________________________________________________
dense_2 (Dense)              (None, 2)                 4098
=================================================================
Total params: 23,591,810
Trainable params: 4,098
Non-trainable params: 23,587,712


"""
