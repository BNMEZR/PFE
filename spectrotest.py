# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 20:28:49 2020

@author: Lenovo
"""
import pandas as pd
#from pandas import DataFrame
import numpy as np
#from numpy import  array, moveaxis, indices, dstack
import matplotlib.pyplot as plt
import librosa
import librosa.display
#import split_folders
#import IPython.display
#import random
import warnings
import os
#from PIL import Image
import pathlib
#import csv
# sklearn Preprocessing
#from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#Keras
#import keras
#import pydot
#from keras.utils.vis_utils import model_to_dot
#keras.utils.vis_utils.pydot = pydot
from tensorflow.keras.utils import plot_model

warnings.filterwarnings('ignore')
#from keras import layers
from tensorflow.python.keras.layers import Activation, Dense, Dropout, Conv2D, Flatten, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling1D, AveragePooling2D, Input, Add
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD

#import csv
"""generating dataspectrogram """

dataset = 'pain painF discomforta discomfortaF discomfortb discomfortbF discomfortc discomfortcF '.split()
for g in dataset:
    pathlib.Path(f'img_data/{g}').mkdir(parents=True, exist_ok=True)
    for filename in os.listdir(f'./dataset/{g}'):
        cryname = f'./dataset/{g}/{filename}'
        y, sr = librosa.load(cryname, mono=True, duration=10, sr=48000)
        print(y.shape)
        plt.specgram(y, NFFT=2048, Fs=512, Fc=0, noverlap=128, sides='default', mode='default', scale='dB');
        plt.axis('off');      
        plt.savefig(f'img_data/{g}/{filename[:-3].replace(".", "")}.png')
        plt.clf()
        
    
      #  """split the data into 80% TRAIN 20% TEST """
         #split_folders.ratio('./img_data/', output="./data", seed=1337, ratio=(.8, .2))
       
        """generating new data train_test"""
        
        train_datagen = ImageDataGenerator(
        rescale=1./255, # rescale all pixel values from 0-255, so aftre this step all our pixel values are in range (0,1)
        shear_range=0.2, #to apply some random tranfromations
        zoom_range=0.2, #to apply zoom
        horizontal_flip=True) # image will be flipper horiz
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        './data/train',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical',
        shuffle = False)
augmented_images = [training_set[0][0][0] for i in range(5)]

test_set = test_datagen.flow_from_directory(
        './data/val',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical',
        shuffle = False )
augmented_images = [test_set[0][0][0] for i in range(5)]

            # """ CNN """
             
model = Sequential()
input_shape=(64, 64, 3)
#1st hidden layer
model.add(Conv2D(32, (3, 3), strides=(2, 2), input_shape=input_shape))
model.add(AveragePooling2D((2, 2), strides=(2,2)))
model.add(Activation('relu'))
#2nd hidden layer
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(AveragePooling2D((2, 2), strides=(2,2)))
model.add(Activation('relu'))
#3rd hidden layer
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(AveragePooling2D((2, 2), strides=(2,2)))
model.add(Activation('relu'))
#Flatten
model.add(Flatten())
model.add(Dropout(rate=0.5))
#Add fully connected layer.
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(rate=0.5))
#Output layer
model.add(Dense(8))
model.add(Activation('softmax'))
model.summary()
#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

   # """ SGD learning training"""
epochs = 100
batch_size = 8
learning_rate = 0.01
decay_rate = learning_rate / epochs
momentum = 0.9
sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=['accuracy'])

# using 100 epoch testing 
history= model.fit (
training_set,
steps_per_epoch=len(training_set) // batch_size,
epochs=100,
validation_data=test_set,
validation_steps=len(test_set) // batch_size),

  # plot accuracy/error for training and validation

#Model Evaluation
score= model.evaluate_generator(generator=test_set, steps=50)
print("loss: %.3f - acc: %.3f" % (score[0], score[1]))
test_set.reset()
pred = model.predict_generator(test_set, steps=50, verbose=1)
#predict

predicted_class_indices=np.argmax(pred,axis=1)
labels = (training_set.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
predictions = predictions[:200]
filenames=test_set.filenames
print(len(filename), len(predictions))
#csv
results=pd.DataFrame.from_dict({"Filename":filenames,"Predictions":predictions},orient='index' )
results.to_csv("prediction_results.csv",index=False)
model.save('64x3-CNN.model')

