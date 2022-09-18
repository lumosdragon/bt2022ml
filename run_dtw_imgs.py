'''
Created on 17 June 2022

ML model for UER baselining

Run network using CNN, where the input are images

@author : Lykourgos Kekempanos
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from pyts.metrics import dtw
from pyts.metrics.dtw import (cost_matrix, accumulated_cost_matrix,
                              _return_path, _blurred_path_region)

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
    Y = data['base_uer_trace'].map(lambda x:  unfold_str_v1(x))
    
    
    
    print("create images")
    
    for idx, val in data.iterrows():
        #print("image no. ", idx)
        plt.ion()
        base_trace = X[idx]  
        dem_trace = Y[idx]
        X_size = len(base_trace)
        Y_size = len(dem_trace)
        mask = np.ones(X_size)
        mask[::5] = 0
        dem_trace
        timestamps_1 = np.arange(X_size +1)
        timestamps_2 = np.arange(Y_size +1)
        dtw_classic, path_classic = dtw(base_trace, dem_trace, dist = 'square',
                                        method = 'classic', return_path = True)
        matrix_classic = np.zeros((X_size, Y_size))
        matrix_classic[tuple(path_classic)] = 1
        plt.pcolor(timestamps_1, timestamps_2, matrix_classic.T,
                   edgecolors='k', cmap='viridis')
        plt.title("{0}\nDTW(x, y) = {1:.2f}".format('classic', dtw_classic),
          fontsize=10)
        plt.ioff()
        plt.savefig((str(p) + "/imgs/dtw/" +str( idx) + ".png"))

        
    print("create images completed ✓")
    
    