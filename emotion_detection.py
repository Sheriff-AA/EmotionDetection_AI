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


# IMAGE VISUALISATION

# Plot a random image from the dataset along with facial keypoints
# Image data is plotted using imshow
# 15 x and y coordinates for each image
# x co-ordinates are in even columns, and y co-ordinates are in odd columns

i = np.random.randint(1, len(dataset1))
plt.imshow(dataset1["Image"][i], cmap="gray")
for j in range(1, 31, 2):
    plt.plot(dataset1.loc[i][j-1], dataset1.loc[i][j], "rx")
    
# Viweing more images in grid format
fig = plt.figure(figsize=(20, 20))

for i in range(16):
    fig.add_subplot(4, 4, i+1)
    plt.imshow(dataset1["Image"][i], cmap="gray")
    for j in range(1, 31, 2):
        plt.plot(dataset1.loc[i][j-1], dataset1.loc[i][j], "rx")  # "rx" = red colour, x
        

# SANITY CHECK - Randomly visualising 64 images with their corresponding key points
fig2 = plt.figure(figsize=(20, 20))

for i in range(64):
    fig2.add_subplot(8, 8, i+1)
    j = np.random.randint(1, len(dataset1))
    plt.imshow(dataset1["Image"][j], cmap="gray")
    for k in range(1, 31, 2):
        plt.plot(dataset1.loc[j][k-1], dataset1.loc[j][k], "rx")


# IMAGE AUGMENTATION

dataset1_copy = copy.copy(dataset1)

columns = dataset1_copy.columns[:-1]

# Horizontal flip - Flip Images along y axis
dataset1_copy["Image"] = dataset1_copy["Image"].apply(lambda x: np.flip(x, axis = 1))

for i in range(len(columns)):
    if i%2 == 0:
        dataset1_copy[columns[i]] = dataset1_copy[columns[i]].apply(lambda x: 96. - float(x))
