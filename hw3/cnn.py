import numpy as np
# import pandas as pd
import csv
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.models import load_model
import sys
import os
import argparse
from random import shuffle
import itertools
import math
from math import log, floor
import random





def _shuffle(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])

def split_valid_set(X_all, Y_all, percentage):
    all_data_size = len(X_all)
    valid_data_size = int(floor(all_data_size * percentage))

    X_all, Y_all = _shuffle(X_all, Y_all)

    X_valid, Y_valid = X_all[0:valid_data_size], Y_all[0:valid_data_size]
    X_train, Y_train = X_all[valid_data_size:], Y_all[valid_data_size:]

    return X_train, Y_train, X_valid, Y_valid  


# parser = argparse.ArgumentParser(description='train an cnn')
# parser.add_argument('--outdir', type=str, default='output')
# args = parser.parse_args()


# for reproduced
# np.random.seed(0)
random.seed(0)

# load training data 

# path_data = os.environ.get('GRAPE_DATASET_DIR')
# input_path = os.path.join(path_data, 'train.csv')

data_train = []
label_train = []
with open(sys.argv[1]) as img:  #'train.csv'input_path
    next(img)

    for row in img:
        label , data = row.strip().split(',')
        data_train.append(data.split(' '))
        label_train.append(label)
data_train = np.array(data_train,dtype=np.float)
data_train = (data_train-data_train.mean())/data_train.std() # normalize
label_train = np.array(label_train,dtype=np.int)

# data_train, label_train, data_val, label_val = split_valid_set(data_train, label_train, 0.15)

# modify to input data shape

data_train = data_train.reshape(data_train.shape[0],48,48,1)
# data_val = data_val.reshape(data_val.shape[0],48,48,1)
input_shape = (48,48,1)

# convert label to one-hot form

label_train = np_utils.to_categorical(label_train)
# label_val = np_utils.to_categorical(label_val)

# datagen = ImageDataGenerator(
#         featurewise_center=True,
#         featurewise_std_normalization=True,
#         rotation_range=20,
#         width_shift_range=0.2,
#         height_shift_range=0.2,
#         horizontal_flip=True)

# datagen.fit(data_train)

# build model
# cnn 512 dropout 0.3->0.5


model = Sequential()
# model.add(Conv2D(32,kernel_size=(3,3),padding='same',activation='relu',input_shape=input_shape))
# model.add(Dropout(0.3))
# model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2))) 

model.add(Conv2D(64,kernel_size=(3,3),padding='same',activation='relu',input_shape=input_shape))
model.add(Dropout(0.3))
model.add(Conv2D(64,kernel_size=(3,3),padding='same',activation='relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2))) 

model.add(Conv2D(128,kernel_size=(3,3),padding='same',activation='relu',input_shape=input_shape))
model.add(Dropout(0.3))
model.add(Conv2D(128,kernel_size=(3,3),padding='same',activation='relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2))) 

# model.add(Conv2D(256,kernel_size=(3,3),padding='same',activation='relu'))
# model.add(Dropout(0.3))
model.add(Conv2D(256,kernel_size=(3,3),padding='same',activation='relu'))
model.add(Dropout(0.3))
model.add(Conv2D(256,kernel_size=(3,3),padding='same',activation='relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

# model.add(Conv2D(256,kernel_size=(3,3),padding='same',activation='relu'))
# model.add(Dropout(0.3))
# model.add(Conv2D(256,kernel_size=(3,3),padding='same',activation='relu'))
# model.add(Dropout(0.3))
# model.add(Conv2D(256,kernel_size=(3,3),padding='same',activation='relu'))
# model.add(Dropout(0.3))
# model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

# model.add(Conv2D(512,kernel_size=(3,3),padding='same',activation='relu'))
# model.add(Dropout(0.3))
model.add(Conv2D(512,kernel_size=(3,3),padding='same',activation='relu'))
model.add(Dropout(0.3))
model.add(Conv2D(512,kernel_size=(3,3),padding='same',activation='relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

# model.add(Conv2D(512,kernel_size=(3,3),padding='same',activation='relu'))
# model.add(Dropout(0.3))
# model.add(Conv2D(512,kernel_size=(3,3),padding='same',activation='relu'))
# model.add(Dropout(0.3))
# model.add(Conv2D(512,kernel_size=(3,3),padding='same',activation='relu'))
# model.add(Dropout(0.3))
# model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dropout(0.5))

# model.add(Dense(1024,activation='relu'))
# model.add(Dropout(0.5))

model.add(Dense(4096,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(7,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adamax',metrics=['accuracy'])
model.fit(data_train, label_train, batch_size=128,epochs=200) #,shuffle=True , validation_split=0.15
# score = model.evaluate(data_val, label_val)

# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

# outdir = args.outdir
# if not os.path.exists(outdir):
#     os.makedirs(outdir)
# output_path = os.path.join(outdir, 'model_56.h5')
model.save('model.h5')  #output_path












