#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 16:34:57 2022

@author: tingys
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pathlib
import glob
import shutil

from pathlib import Path
from PIL import Image

p = Path(__file__).resolve().parents[2]


#----- generate dataset -----#

batch_size = 20
seed = 1880
image_size = (369, 369)
num_classes = 2


datapath = str(p) + "/preprocessed/gafs"

data_dir = str(p) + "/imgs/gafs/"

data_dir = pathlib.Path(data_dir)

image_count = len(list(data_dir.rglob('*.png')))

#----- split dataset into 3 subsets -----#
ds = tf.keras.preprocessing.image_dataset_from_directory(
    datapath,
    image_size=image_size,
    batch_size=None
)



# split for test and train
train_size = int(image_count * 0.8)
train_ds = ds.take(train_size)
test_ds = ds.skip(train_size)

# split for test and validation
val_size = int(train_size * 0.2)
train_ds = train_ds.skip(val_size)
val_ds = train_ds.take(val_size)

print(int(tf.data.experimental.cardinality(train_ds).numpy()))
print(int(tf.data.experimental.cardinality(val_ds).numpy()))
print(int(tf.data.experimental.cardinality(test_ds).numpy()))


#----- buffered prefetching -----#

train_ds = train_ds.prefetch(buffer_size=30)
val_ds = val_ds.prefetch(buffer_size=30)


def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    x = inputs


    x = layers.Conv1D(32, 3, strides = 2, padding = 'same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(3, strides = 2, padding = 'same')(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Dropout(0.2)(x)

    
    x = layers.Flatten()(x)
    
    x = layers.Dropout(0.2)(x)

    
    outputs = layers.Dense(1, activation='softmax')(x)
    return keras.Model(inputs, outputs)


model = tf.keras.Sequential(make_model(input_shape=image_size + (3,), num_classes=2))
keras.utils.plot_model(model, show_shapes=True)

#----- train the model -----#

epochs = 10

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
]
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
hist = model.fit(
    train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds,
)

model.summary()

def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()


plot_hist(hist)