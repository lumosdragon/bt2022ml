#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 14:32:39 2022

@author: tingys
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import glob
import shutil

from pathlib import Path
from PIL import Image

p = Path(__file__).resolve().parents[2]


#----- generate dataset -----#

batch_size = 20
seed = 1880
image_size = (369, 369)

datapath = str(p) + "/preprocessed/gafs"

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    datapath,
    validation_split=0.3,
    subset='training',
    seed=seed,
    image_size=image_size,
    batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    datapath,
    validation_split=0.3,
    subset='validation',
    seed=seed,
    image_size=image_size,
    batch_size=batch_size
)

#----- buffered prefetching -----#

train_ds = train_ds.prefetch(buffer_size=30)
val_ds = val_ds.prefetch(buffer_size=30)

# ----- build network -----#se


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
