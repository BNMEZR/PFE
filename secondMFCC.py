# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 10:24:13 2020

@author: Lenovo
"""

# Load various imports 
import pandas as pd
import os
import librosa
import numpy as np
import importlib_metadata as metadata


file_name='dataset'

def extract_features(file_name):
   
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        mfccsscaled = mfccs - np.mean(mfccs, axis=1).reshape(13, 1)
        
    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None 
     
    return mfccsscaled
    
    

# Set the path to the full UrbanSound dataset 
fulldatasetpath = 'memoireStuff/trial/UrbanSound8K'

metadata = pd.read_csv(fulldatasetpath + '../UrbanSound8K.csv')

features = []

# Iterate through each sound file and extract the features 
for index, row in metadata.iterrows():
    
    file_name = os.path.join(os.path.abspath(fulldatasetpath),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
    
    class_label = row["class_name"]
    data = extract_features(file_name)
    
    features.append([data, class_label])

# Convert into a Panda dataframe 
featuresdf = pd.DataFrame(features, columns=['feature','class_label'])

print('Finished feature extraction from ', len(featuresdf), ' files')