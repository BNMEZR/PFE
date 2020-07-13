# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 12:13:47 2020

@author: Lenovo
"""

import pywt
import scipy.io.wavfile

wavefile = 'spkr09_M_S1a_cry14.wav'
  # read the wavefile
sampling_frequency, signal = scipy.io.wavfile.read(wavefile)
  #
scales = (1, len(signal))
coefficient, frequency = pywt.cwt(signal, scales, 'mexh')
  