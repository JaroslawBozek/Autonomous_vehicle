
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as npimg
import os

## Keras
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense, Lambda, Conv2D, ELU, ReLU, Input
import cv2
import pandas as pd
import random
import ntpath

## Sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

datadir = '/home/tasoth/av_ws/src/av_07/data'
imagedir = 'images/images/'
columns = ['data']
data = pd.read_csv(os.path.join(datadir, 'data.csv'), names = columns)
data = data.drop(0)
data['data']
pd.set_option('display.max_colwidth', -1)
print(data.head())


def allow_memory_growth():
  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
    try:
      # Currently, memory growth needs to be the same across GPUs
      for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
      logical_gpus = tf.config.experimental.list_logical_devices('GPU')
      print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
      # Memory growth must be set before GPUs have been initialized
      print(e)

def load_params(datadir, df):
  image_path = []
  velocity = []
  steering = []
  for i in range(len(data)):
    indexed_data = data.iloc[i]
    center = indexed_data[0]
    splt = center.split(";")
    name = imagedir + splt[0] + ".jpg"
    image_path.append(os.path.join(datadir, name))
    vel = float(splt[1])/120
    steer = float(splt[2])
    velocity.append(vel)
    steering.append(steer)
  image_paths = np.asarray(image_path)
  velocities = np.asarray(velocity)
  steerings = np.asarray(steering)

  return image_paths, velocities, steerings

allow_memory_growth()
image_paths, velocities, steerings = load_params(datadir, data)

param = []
for it, i in enumerate(velocities):
  param.append([velocities[it], steerings[it]])
params = np.asarray(param)

def img_preprocess(img):
  img = npimg.imread(img)
  img = img[380:600, :, :]
  img = cv2.resize(img, (200, 200))
  return img

X_train, X_valid, Y_train, Y_valid = train_test_split(image_paths, params, test_size=0.2, random_state=0)
X_train = np.array(list(map(img_preprocess, X_train)))
X_valid = np.array(list(map(img_preprocess, X_valid)))


def nvidia_model():
  model = Sequential()
  model.add(Input(shape=(200, 200, 3)))

  # Five convolutional layer with dropout
  model.add(Conv2D(24, 5, strides=(2, 2), padding="valid", kernel_initializer='glorot_uniform'))
  model.add(ELU())
  model.add(Dropout(0.2))
  model.add(Conv2D(36, 5, strides=(2, 2), padding="valid", kernel_initializer='glorot_uniform'))
  model.add(ELU())
  model.add(Dropout(0.2))
  model.add(Conv2D(48, 5, strides=(2, 2), padding="valid", kernel_initializer='glorot_uniform'))
  model.add(ELU())
  model.add(Dropout(0.2))
  model.add(Conv2D(64, 3, strides=(1, 1), padding="valid", kernel_initializer='glorot_uniform'))
  model.add(ELU())
  model.add(Dropout(0.2))
  model.add(Conv2D(64, 3, strides=(1, 1), padding="valid", kernel_initializer='glorot_uniform'))
  model.add(ELU())
  model.add(Dropout(0.2))
  # Flatten layer
  model.add(Flatten())
  # Three fully connected layer with dropout on first two
  model.add(Dense(200, kernel_initializer='glorot_uniform'))
  model.add(ELU())
  model.add(Dropout(0.2))
  model.add(Dense(100, kernel_initializer='glorot_uniform'))
  model.add(ELU())
  model.add(Dropout(0.2))
  model.add(Dense(50, kernel_initializer='glorot_uniform'))
  model.add(ELU())
  model.add(Dense(2, kernel_initializer='glorot_uniform'))
  optimizer = Adam(lr=1e-3)
  model.compile(loss='mse', optimizer=optimizer, metrics=['Accuracy'])
  return model




model = nvidia_model()
print(model.summary())



print(Y_train)
# history = model.fit(X_train, Y_train, epochs=5, validation_data=(X_valid, Y_valid), batch_size=128, verbose=1, shuffle=1)
history = model.fit(X_train, Y_train,
                    batch_size=32,
                    epochs=8,
                    verbose=1,
                    validation_data=(X_valid, Y_valid))

model.save('/home/tasoth/av_ws/src/av_07/scripts/model10.model')


