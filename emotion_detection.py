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
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import copy
import os
import PIL
import seaborn as sns
import pickle
from PIL import *
import random



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
dataset1_copy = copy.deepcopy(dataset1)
columns = dataset1_copy.columns[:-1]

# Horizontal flip - Flip Images along y axis
dataset1_copy["Image"] = dataset1_copy["Image"].apply(lambda x: np.flip(x, axis = 1))

# y coordinate values would be the same
for i in range(len(columns)):
    if i%2 == 0:
        dataset1_copy[columns[i]] = dataset1_copy[columns[i]].apply(lambda x: 96. - float(x))

# Show the Original Image
plt.imshow(dataset1["Image"][0], cmap="gray")
for j in range(1, 31, 2):
    plt.plot(dataset1.loc[0][j-1], dataset1.loc[0][j], "rx")


# Show the Horizontally flipped image
plt.imshow(dataset1_copy["Image"][0], cmap="gray")
for j in range(1, 31, 2):
    plt.plot(dataset1_copy.loc[0][j-1], dataset1_copy.loc[0][j], "rx")


# Concatenate the original dataframe with augmented dataframe
augmented_ds = np.concatenate((dataset1, dataset1_copy))
print(augmented_ds.shape)


# Increasing brightness of images
dataset1_copy = copy.deepcopy(dataset1)
dataset1_copy["Image"] = dataset1["Image"].apply(lambda x:np.clip(random.uniform(1.5, 2)* x, 0.0, 255.0))
augmented_ds = np.concatenate((augmented_ds, dataset1_copy))
print(augmented_ds.shape)

plt.imshow(dataset1_copy["Image"][0], cmap="gray")
for j in range(1, 31, 2):
    plt.plot(dataset1_copy.loc[0][j-1], dataset1_copy.loc[0][j], "rx")


# MINI CHALLENGE 3 & 4 
# Flipping images vertically
dataset1_copy = copy.deepcopy(dataset1)
columns = dataset1_copy.columns[:-1]

dataset1_copy["Image"] = dataset1_copy["Image"].apply(lambda x: np.flip(x, axis=0))

for i in range(len(columns)):
    if i%2 != 0:
        dataset1_copy[columns[i]] = dataset1_copy[columns[i]].apply(lambda x: 96 - float(x))

# SANITY CHECK
plt.imshow(dataset1_copy["Image"][0], cmap="gray")
for j in range(1, 31, 2):
    plt.plot(dataset1_copy.loc[0][j-1], dataset1_copy.loc[0][j], "rx")


# PERFORM DATA NORMALIZATION AND SPLITTING INTO TRAINING AND TEST SET
# Obtaining the image values in the 31st column
images = augmented_ds[:,30]

# Normalize the images
images = images/255
X = np.empty((len(images), 96, 96, 1))  # creating an empty array of 6420 rows, each will be 96 by 96 by 1

for i in range(len(images)):
    X[i,] = np.expand_dims(images[i], axis=2)

# convert array to float32
X = np.asarray(X).astype(np.float32)
print(X.shape)

y = augmented_ds[:,:30]
y = np.asarray(y).astype(np.float32)    # converting to float 32
print(y.shape)

# Splitting the data into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



















