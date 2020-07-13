# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 11:47:44 2020

@author: Lenovo
"""
import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt
import pywt
import os
import pathlib


FIG_SIZE = (15,10)
# load audio file with Librosa

dataset = 'pain discomforta '.split()
for g in dataset:
    pathlib.Path(f'featuredwt/{g}').mkdir(parents=True, exist_ok=True)
    for filename in os.listdir(f'./dataset/{g}'):
        cryname = f'./dataset/{g}/{filename}'
        y, sr = librosa.load(cryname, mono=True, duration=10, sr=48000)
        print(y.shape)
discrete_wavelet = pywt.Wavelet('db2')
print(discrete_wavelet)

max_level = pywt.dwt_max_level(len(y), discrete_wavelet)
print('MAXIMUM DECOMPOSE LEVEL = ',max_level)

# decompose
tree = pywt.wavedec(y, 'db2',level=3)
cA3, cD3, cD2, cD1 = tree
#print(len(cD1),len(cD2),len(cD3),len(cA3))

# reconstruct
rec_sample = pywt.waverec(tree, 'db2')
rec_to_orig = pywt.idwt(None, cD1, 'db2', 'smooth')  #
rec_to_level1 = pywt.idwt(None, cD2, 'db2', 'smooth')
rec_to_level2_from_detail = pywt.idwt(None, cD3, 'db2', 'smooth')
rec_to_level2_from_approx = pywt.idwt(cA3, None, 'db2', 'smooth')
#print(len(rec_to_orig),len(rec_to_level1),len(rec_to_level2_from_detail),len(rec_to_level2_from_approx))
plt.axis('off');      
plt.savefig(f'featuredwt/{g}/{filename[:-3].replace(".", "")}.png')
plt.clf()


