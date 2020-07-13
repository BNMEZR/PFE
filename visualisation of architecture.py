# -*- coding: utf-8 -*-
"""
Created on Fri May 15 01:46:15 2020

@author: Lenovo
"""
from ann_visualizer.visualize import ann_viz;
from keras.models import model_from_json
import numpy as np
# fix random seed for reproducibility
np.random.seed(7)
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")
ann_viz(model, title="Artificial Neural network - Model Visualization")