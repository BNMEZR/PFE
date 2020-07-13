# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 15:37:52 2020

@author: Lenovo
"""
#import matplotlib.pyplot as plt
#import os.path 
#src_path = "data/train/discomforta"
#sub_class = os.listdir(src_path)
#
#fig = plt.figure(figsize=(10,5))
#path = os.path.join(src_path,sub_class[0])
#for i in range(4):
#    plt.subplot(240 + 1 + i)
#    img = plt.imread(os.path.join(path))
#    plt.imshow(img, cmap=plt.get_cmap('gray'))
#
#path = os.path.join(src_path,sub_class[1])
#for i in range(4,8):
#    plt.subplot(240 + 1 + i)
#    img = plt.imread(os.path.join(path))
#    plt.imshow(img, cmap=plt.get_cmap('gray'))
from keras.preprocessing.image import ImageDataGenerator,img_to_array, load_img
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
img = load_img('spkr01_M_S1a_cry07.png')
x = img_to_array(img)  # creating a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # converting to a Numpy array with shape (1, 3, 150, 150)
i = 0
for batch in datagen.flow(x,save_to_dir='output', save_prefix='spkr01_M_S1a_cry07', save_format='png'):
    i += 1
    if i > 20:
        break