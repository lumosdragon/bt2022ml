#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 14:32:39 2022

@author: tingys
"""

import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import glob
import shutil

from pathlib import Path

p = Path(__file__).resolve().parents[2]

# train_ds = tf.keras.utils.image_dataset_from_directory(
#     str(p) + "/imgs/gafs",
#     validation_split = 0.3,
#     subset = 'training',
#     seed = 1880,
#     batch_size = 20
    
#     )

datapath = str(p) + "/preprocessed/input_dataset_simple.csv"

#read file
data = pd.read_csv(datapath)
y = data['copper_outcome_mapping_binary'] # 1: c_ok, 0: c_dis
gafs_1 = (str(p) + '/preprocessed/gafs/gafs_1')
gafs_0 = (str(p) + '/preprocessed/gafs/gafs_0')


for filename in glob.glob(str(p)+ "/imgs/gafs/*.png"):
     #s = filename
     
     filename = str(filename)
     src = str(filename)
     filename = filename.replace("_"," ")
     filename = filename.replace(".png"," str_end")
     img_idx = [int(s) for s in filename.split() if s.isdigit()][0]
     if y[img_idx] == 1:
         shutil.copy(src, gafs_1)
     else:
         shutil.copy(src, gafs_0)
     
     
     