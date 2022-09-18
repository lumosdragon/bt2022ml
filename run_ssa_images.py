#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 03:17:36 2022

@author: tingys
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from pyts.decomposition import SingularSpectrumAnalysis

import glob
from PIL import Image
from os import path

def unfold_str_v1(x): 
    y = x.replace("[","").replace("]","").split(',')
    return [float(y[i]) for i in range(len(y))]

if __name__ == '__main__':
    
    #========================================================================
    #Load training data
    #========================================================================
    
    #two classes c_ok and c_dis
    nb_classes = 2
    
    #get data file path 
    p = Path(__file__).resolve().parents[2]
    #datapath = str(p) + "/data/input_dataset_for_MLP_union.csv"
    #datapath = str(p) + "/data/input_dataset_for_MLP.csv"
    #datapath = str(p) + "/data/input_dataset_for_ML.csv"
    datapath = str(p) + "/preprocessed/input_dataset_simple.csv"
    
    #read file

data = pd.read_csv(datapath)
data.info()

X = data['dem_uer_trace'].map(lambda x:  unfold_str_v1(x))
X.to_frame

x = pd.DataFrame(X.tolist())

temp = x.loc[0]

temp= temp.values.reshape(1,-1)

window_size = 20

groups = 4

ssa = SingularSpectrumAnalysis(window_size = window_size, groups = groups)

x_ssa = ssa.fit_transform(x)

plot_ssa = plt.plot(x_ssa [0, 0], 'o--', label = 'SSA {0}'.format(0))

plot_ssa.legend(loc = 'best', fontsize =14)
    
