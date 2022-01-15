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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import copy
import os
import PIL
import seaborn as sns
import pickle
from PIL import *
import random
import tensorflow as tf
from tensorflow import keras
import cv2
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from IPython.display import display
from tensorflow.python.keras import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, optimizers
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from keras import optimizers
import json
from keras.utils import to_categorical

#%%
def res_block(x, filter, stage):
    
    # CONVOLUTION BLOCK
    x_copy = x
    f1, f2, f3 = filter
    
    
    # Main Path
    x = Conv2D(f1, (1, 1), strides=(1, 1), name="res_"+str(stage)+"_conv_a", kernel_initializer=glorot_uniform(seed=0))(x)
    x = MaxPool2D((2,2))(x)
    x = BatchNormalization(axis=3, name="bn_"+str(stage)+"_conv_a")(x)
    x = Activation("relu")(x)
    
    x = Conv2D(f2, kernel_size=(3, 3), strides=(1, 1), padding="same", name="res_"+str(stage)+"_conv_b", kernel_initializer=glorot_uniform(seed=0))(x)   
    x = BatchNormalization(axis=3, name="bn_"+str(stage)+"_conv_b")(x)
    x = Activation("relu")(x)
    
    x = Conv2D(f3, kernel_size=(1, 1), strides=(1, 1), name="res_"+str(stage)+"_conv_c", kernel_initializer=glorot_uniform(seed=0))(x)   
    x = BatchNormalization(axis=3, name="bn_"+str(stage)+"_conv_c")(x)
    
    
    # Short Path
    x_copy = Conv2D(f3, kernel_size=(1, 1), strides=(1, 1), name="res_"+str(stage)+"_conv_copy", kernel_initializer=glorot_uniform(seed=0))(x_copy)
    x_copy = MaxPool2D((2,2))(x_copy)
    x_copy = BatchNormalization(axis=3, name="bn_"+str(stage)+"_conv_copy")(x_copy)
    
    # Add
    x = Add()([x, x_copy])
    x = Activation("relu")(x)
    
    # Identity block 1
    x_copy = x
    
    # Main path
    x = Conv2D(f1, (1, 1), strides=(1, 1), name="res_"+str(stage)+"_identity_1_a", kernel_initializer=glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3, name="bn_"+str(stage)+"_identity_1_a")(x)
    x = Activation("relu")(x)
    
    x = Conv2D(f2, kernel_size=(3, 3), strides=(1, 1), padding="same", name="res_"+str(stage)+"_identity_1_b", kernel_initializer=glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3, name="bn_"+str(stage)+"_identity_1_b")(x)
    x = Activation("relu")(x)
    
    x = Conv2D(f3, kernel_size=(1, 1), strides=(1, 1), name="res_"+str(stage)+"_identity_1_c", kernel_initializer=glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3, name="bn_"+str(stage)+"_identity_1_c")(x)
    
    # Add
    x = Add()([x, x_copy])
    x = Activation("relu")(x)
    
    # Identity block 2
    x_copy = x
    
    # Main path
    x = Conv2D(f1, (1, 1), strides=(1, 1), name="res_"+str(stage)+"_identity_2_a", kernel_initializer=glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3, name="bn_"+str(stage)+"_identity_2_a")(x)
    x = Activation("relu")(x)
    
    x = Conv2D(f2, kernel_size=(3, 3), strides=(1, 1), padding="same", name="res_"+str(stage)+"_identity_2_b", kernel_initializer=glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3, name="bn_"+str(stage)+"_identity_2_b")(x)
    x = Activation("relu")(x)
    
    x = Conv2D(f3, kernel_size=(1, 1), strides=(1, 1), name="res_"+str(stage)+"_identity_2_c", kernel_initializer=glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3, name="bn_"+str(stage)+"_identity_2_c")(x)
    
    # Add
    x = Add()([x, x_copy])
    x = Activation("relu")(x)
    
    return x

#%%


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

#%%

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


#%%
# IMAGE AUGMENTATION
dataset1_copy = copy.deepcopy(dataset1)
columns = dataset1_copy.columns[:-1]

# Horizontal flip - Flip Images along y axis
dataset1_copy["Image"] = dataset1_copy["Image"].apply(lambda x: np.flip(x, axis = 1))

# y coordinate values would be the same
for i in range(len(columns)):
    if i%2 == 0:
        dataset1_copy[columns[i]] = dataset1_copy[columns[i]].apply(lambda x: 96. - float(x))
#%%
# Show the Original Image
plt.imshow(dataset1["Image"][0], cmap="gray")
for j in range(1, 31, 2):
    plt.plot(dataset1.loc[0][j-1], dataset1.loc[0][j], "rx")

#%%
# Show the Horizontally flipped image
plt.imshow(dataset1_copy["Image"][0], cmap="gray")
for j in range(1, 31, 2):
    plt.plot(dataset1_copy.loc[0][j-1], dataset1_copy.loc[0][j], "rx")

#%%
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

#%%
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

#%%

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


#%%
# BUILD DEEP RESIDUAL NEURAL NETWORK KEY FACIAL POINTS DETECTION MODEL


input_shape = (96, 96, 1)

# Input tensor shape
x_input = Input(input_shape)

# Zero-padding
x = ZeroPadding2D((3, 3))(x_input)

# Stage 1
x = Conv2D(64, (7, 7), strides=(2, 2), name="conv1", kernel_initializer=glorot_uniform(seed=0))(x)
x = BatchNormalization(axis=3, name="bn_conv1")(x)
x = Activation("relu")(x)
x = MaxPooling2D((3, 3), strides=(2, 2))(x)

# Stage 2
x = res_block(x, filter=[64, 64, 256], stage=2)

# Stage 3
x = res_block(x, filter=[128, 128, 512], stage=3)

# Average Pooling
x = AveragePooling2D((2, 2), name="Averagea_Pooling")(x)

# Final Layer
x = Flatten()(x)
x = Dense(4096, activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(2048, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(30, activation="relu")(x)


model_1_facialkeypoints = Model(inputs=x_input, outputs=x)
model_1_facialkeypoints.summary()

#%%

# COMPILE AND TRAIN KEY FACIAL POINTS DETECTOR MODEL
adam = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model_1_facialkeypoints.compile(loss="mean_squared_error", optimizer=adam, metrics=["accuracy"])

# save the best method with the least validation loss
checkpointer = ModelCheckpoint(filepath="FacialKeyPoints_weights.hdf5", verbose=1, save_best_only=True)

history = model_1_facialkeypoints.fit(X_train, y_train, batch_size=64, epochs=130, validation_split=0.10, callbacks=[checkpointer])

# save the model architecture to json file for future use
model_json = model_1_facialkeypoints.to_json()
with open("FacialKeyPoints_Model.json", "w") as json_file:
    json_file.write(model_json)


#%%

# ASSESS TRAINED RES-NET MODEL PERFORMANCE
with open("FacialKeyPoints_Model.json", "r") as json_file:
    json_savedmodel = json_file.read()

# Load Model Architecture
model_1_facialkeypoints = tf.keras.models.model_from_json(json_savedmodel)
model_1_facialkeypoints.load_weights("FacialKeyPoints_weights.hdf5")
adam = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model_1_facialkeypoints.compile(loss="mean_squared_error", optimizer=adam, metrics=["accuracy"])

# Evaluate Model
result = model_1_facialkeypoints.evaluate(X_test, y_test)
print("ACCURACY : {}".format(result[1]))

# Model Keys
history.history.keys()

# Plot the training artifacts
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["train_loss", "val_loss"], loc="upper right")
plt.show()


#%%
# IMPORT AND EXPLORE DATASET FOR FACIAL EXPRESSION DETECTION
face_expression_dataset = pd.read_csv("icml_face_data.csv")
face_expression_dataset.head()
face_expression_dataset[" pixels"][0]

def string2array(x):
    return np.array(x.split(" ")).reshape(48, 48, 1).astype("float32")

# RESIZE IMAGES FROM (48x48) to (96x96)
def resize(x):
    img = x.reshape(48, 48)
    return cv2.resize(img, dsize=(96, 96), interpolation=cv2.INTER_CUBIC)

face_expression_dataset[" pixels"] = face_expression_dataset[" pixels"].apply(lambda x: string2array(x))
face_expression_dataset[" pixels"] = face_expression_dataset[" pixels"].apply(lambda x: resize(x))

face_expression_dataset.head()
face_expression_dataset.shape

label_to_text = {0:"Anger", 1:"Disgust", 2:"Sad", 3:"Happiness", 4:"Surprise"}

plt.imshow(face_expression_dataset[" pixels"][0], cmap="gray")

#%%
# VISUALIZE IMAGE AND PLOT LABELS
emotions = [1, 2, 3, 4]

for i in emotions:
    data = face_expression_dataset[face_expression_dataset["emotion"] == i][:1]
    img = data[" pixels"].item()
    img = img.reshape(96, 96)
    plt.figure()
    plt.title(label_to_text[i])
    plt.imshow(img, cmap="gray")

print(face_expression_dataset.emotion.value_counts())
plt.figure(figsize=(10,10))
sns.barplot(x=face_expression_dataset.emotion.value_counts().index, y=face_expression_dataset.emotion.value_counts())

#%%
# DATA PREPARATION AND IMAGE AUGMENTATION

# split dataframe into features and models
X = face_expression_dataset[" pixels"]
y = to_categorical(face_expression_dataset["emotion"])

X = np.stack(X, axis=0)
X = X.reshape(24568, 96, 96, 1)

print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, shuffle=True)

print(X_val.shape, y_val.shape)

# image pre-processing
X_train = X_train/255
X_val = X_val/255
X_test = X_test/255

#%%
data_gen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1,
                                   height_shift_range=0.1, shear_range=0.1,
                                   zoom_range=0.1, horizontal_flip=True,
                                   fill_mode="nearest", vertical_flip=True,
                                   brightness_range=[1.1, 1.5])
#%%
input_shape = (96, 96, 1)

# Input tensor shape
X_input = Input(input_shape)

# Zero-padding
X = ZeroPadding2D((3, 3))(X_input)

# Stage 1
X = Conv2D(64, (7, 7), strides=(2, 2), name="conv1", kernel_initializer=glorot_uniform(seed=0))(X)
X = BatchNormalization(axis=3, name="bn_conv1")(X)
X = Activation("relu")(X)
X = MaxPooling2D((3, 3), strides=(2, 2))(X)

# Stage 2
X = res_block(X, filter=[64, 64, 256], stage=2)

# Stage 3
X = res_block(X, filter=[128, 128, 512], stage=3)

# Stage 4
# X = res_block(X, filter= [256, 256, 1024], stage= 4)

# Average Pooling
X = AveragePooling2D((4, 4), name="Averagea_Pooling")(X)

# Final layer
X = Flatten()(X)
X = Dense(5, activation="softmax", name="Dense_final", kernel_initializer=glorot_uniform(seed=0))(X)

model_2_emotion = Model(inputs=X_input, outputs = X, name="Resnet18")
model_2_emotion.summary()

#%%
# Train the network
model_2_emotion.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])

#Categorical cross entropy because we have 5 categories

#%%
earlystopping = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=20)

# Save the best model with lower validation loss
checkpointer = ModelCheckpoint(filepath="FacialExpression_weights.hdf5", verbose=1, save_best_only=True)
history = model_2_emotion.fit(data_gen.flow(X_train, y_train, batch_size=64),
                             validation_data=(X_val, y_val), steps_per_epoch=len(X_train)//64,
                             epochs=2, callbacks=[checkpointer, earlystopping])

# set epochs to 100 or more

# saving the model architecture to json file for future use
model_json = model_2_emotion.to_json()
with open("FacialExpression_model.json", "w") as json_file:
    json_file.write(model_json)

#%%    
# ASSESS PERFORMANCE OF TRAINED FACIAL EXPRESSION CLASSIFIER
with open("FacialExpression_model.json", "r") as json_file:
    json_savedmodel = json_file.read()

# Load the model Architecture
model_2_emotion = tf.keras.models.model_from_json(json_savedmodel)
model_2_emotion.load_weights("weights_emotions.hdf5")
model_2_emotion.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])

score = model_2_emotion.evaluate(X_test, y_test)
print("Test Accuracy: {}".format(score[1]))

print(history.history.keys())

accuracy = history.history["acc"]
val_accuracy = history.history["val_acc"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]


#%%
epochs = range(len(accuracy))

plt.plot(epochs, accuracy, "bo", label="Training Accuracy")
plt.plot(epochs, val_accuracy, "b", label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()

plt.plot(epochs, loss, "ro", label="Training Loss")
plt.plot(epochs, val_loss, "r", label="Validation Loss")
plt.title("Training and Validation Loss")
plt.legend()


predicted_classes = np.argmax(model_2_emotion.predict(X_test), axis=-1)
y_true = np.argmax(y_test, axis=-1)
y_true.shape

cm = confusion_matrix(y_true, predicted_classes)
plt.figure(figsize = (10, 10))
sns.heatmap(cm, annot = True, cbar = False)

#%%
# Print out a grid of 25 images along with their predicted/true label
# Print out the classification report and analyze precision and recall
L = 5
W = 5

fig, axes = plt.subplots(L, W, figsize = (24, 24))
axes = axes.ravel()

for i in np.arange(0, L*W):
    axes[i].imshow(X_test[i].reshape(96,96), cmap = 'gray')
    axes[i].set_title('Prediction = {}\n True = {}'.format(label_to_text[predicted_classes[i]], label_to_text[y_true[i]]))
    axes[i].axis('off')

plt.subplots_adjust(wspace = 1)   
print(classification_report(y_true, predicted_classes))


#%%
# COMBINING BOTH FACIAL EXPRESSIONS
def predict(X_test):
    
    #make prediction from keypoint model
    df_predict = model_1_facialkeypoints.predict(X_test)
    #prediction from emotion model
    df_emotion = np.argmax(model_2_emotion.predict(X_test), axis=1)
    #reshaping array from (856,) to (856,1)
    df_emotion=np.expand_dims(df_emotion, axis=1)
    #pedictions to dataframe
    df_predict = pd.DataFrame(df_predict, columns=columns)
    #adding emotion to dataframe
    df_predict["emotion"] = df_emotion
    return df_predict

df_predict = predict(X_test)
print(df_predict.head())


fig, axes = plt.subplots(4, 4, figsize=(24, 24))
axes = axes.ravel()

for i in range(16):
    axes[i].imshow(X_test[i].squeeze(), cmap="gray")
    axes[i].set_title("Prediction = {}".format(label_to_text[df_predict["emotion"][i]]))
    axes[i].axis("off")
    for j in range(1,31,2):
        axes[i].plot(df_predict.loc[i][j-1], df_predict.loc[i][j], "rx")

#%%

def deploy(directory, model):
    MODEL_DIR = directory
    version = 1
    
    #joining temp model directory with chosen version number
    export_path = os.path.join(MODEL_DIR, str(version))
    print("export_path = {}\n".format(export_path))
    
    #save model using save_model.save
    # if dir exists, remove it
    
    if os.path.isdir(export_path):
        print("\nAlready saved a model, cleaning up\n")
        # !rm -r{export_path}
        
    tf.saved_model.save(model, export_path)
    os.environ["MODEL_DIR"]
    
#%%

# add tensorflow-model-server package to list of packages 
# !echo "deb http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | tee /etc/apt/sources.list.d/tensorflow-serving.list && \
# curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | apt-key add -
# !apt update
#%%

#install tensorflow serving
# !apt-get install tensorflow-model-server
    
    
