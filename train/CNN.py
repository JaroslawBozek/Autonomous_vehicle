
import numpy as np
import argparse
import matplotlib.image as npimg
import os

## Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense, Lambda, Conv2D, ELU, ReLU, Input
import cv2
import pandas as pd

## Sklearn
from sklearn.model_selection import train_test_split


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
      
      
def load_params(imagedir, data):
  image_path = []
  velocity = []
  steering = []
  for i in range(len(data)):
    indexed_data = data.iloc[i]
    center = indexed_data[0]
    splt = center.split(";")
    name = splt[0] + ".jpg"
    image_path.append(os.path.join(imagedir, name))
    vel = float(splt[1])/120
    steer = float(splt[2])
    velocity.append(vel)
    steering.append(steer)
  image_paths = np.asarray(image_path)
  velocities = np.asarray(velocity)
  steerings = np.asarray(steering)

  return image_paths, velocities, steerings


def img_preprocess(img):
  img = npimg.imread(img)
  img = img[380:600, :, :]
  img = cv2.resize(img, (200, 200))
  return img


def main(arg):
  imagedir = arg.images_path
  csvdir = arg.labels_path
  columns = ['data']
  data = pd.read_csv(os.path.join(csvdir, arg.labels_name), names = columns)
  data = data.drop(0)
  data['data']
  pd.set_option('display.max_colwidth', -1)
  print(data.head())
  allow_memory_growth()
  image_paths, velocities, steerings = load_params(imagedir, data)

  param = []
  for it, i in enumerate(velocities):
    param.append([velocities[it], steerings[it]])
  params = np.asarray(param)


  X_train, X_valid, Y_train, Y_valid = train_test_split(image_paths, params, test_size=0.2, random_state=0)
  X_train = np.array(list(map(img_preprocess, X_train)))
  X_valid = np.array(list(map(img_preprocess, X_valid)))

  model = nvidia_model()
  print(model.summary())
  history = model.fit(X_train, Y_train,
                      batch_size=32,
                      epochs=8,
                      verbose=1,
                      validation_data=(X_valid, Y_valid))
  print(history)
  model.save(arg.model_path + arg.model_name + '.model')

  
if __name__ == "__main__":
    allow_memory_growth()

    parser = argparse.ArgumentParser()
    parser.add_argument('--images_path', type=str, help='Path to train images', required=True)
    parser.add_argument('--labels_path', type=str, help='Path to csv file', required=True)
    parser.add_argument('--labels_name', type=str, help='Name of csv file', required=True)
    parser.add_argument('--model_path', type=str, help='Path to save the model', required=True)
    parser.add_argument('--model_name', type=str, default='model', help='Name of the generated model', required=False)

    args, _ = parser.parse_known_args()
    main(args)
