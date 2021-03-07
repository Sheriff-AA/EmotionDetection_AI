# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 19:32:44 2021

@author: SHERIF ATITEBI O
"""

# FIRST MODEL
# DETECT KEY FACIAL POINTS
# input images are 96*96

# IMPORTING LIBRARIES
import pandas as pd
import numpy as np
import os
import PIL
# import seaborn as sns
import pickle
from PIL import *

# IMPORTING DATASET
dataset1 = pd.read_csv("data.csv")

print(dataset1.info())       # information about the dataset
print(dataset1.isnull().sum())     # do null values exist in the dataset?


# the image column of our dataset is a 2140*1 vector (1D array)
# each row of the "Image" column consists of the image information/values
# it is necessary to transform it into a numpy array 
# then convert it to a 2D array of shape 96*96, because that is the size
# of the actual image

#print(dataset1["Image"][0])

dataset1["Image"] = dataset1["Image"].apply(lambda x: np.fromstring(x, dtype=int, sep=" ").reshape(96, 96))

#print(dataset1["Image"][0])

# obtaining the avg, mim, and max of column "right_eye_center_x"
print(dataset1.describe())

print(dataset1["right_eye_center_x"].max())
print(dataset1["right_eye_center_x"].min())
print(dataset1["right_eye_center_x"].sum()/ len(dataset1["right_eye_center_x"]))
