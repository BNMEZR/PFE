# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 11:46:16 2020

@author: Lenovo
"""

import os
import sys
import types
import boto3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from io import BytesIO
from zipfile import ZipFile
from botocore.client import Config

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


import soundfile as sf
import librosa

import pywt