# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 14:06:11 2020

@author: Lenovo
"""

import librosa
file ="spkr09_M_S1a_cry14.wav"
data, sample_rate = librosa.load(file, sr=None)
print(data.shape, data)
print(sample_rate)
