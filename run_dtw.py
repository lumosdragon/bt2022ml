'''
Candidate Number: 29441
This code runs the Dynamic Time Warp (DTW)
algorithm.
Majority of this code is written by the candidate,
referencing the 'pyts' package for running the 
DTW model at 
https://pyts.readthedocs.io/en/stable/auto_examples/metrics/plot_dtw.html#sphx-glr-auto-examples-metrics-plot-dtw-py
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
    
    
    #Load data
    
    #two classes c_ok and c_dis
    nb_classes = 2
    
    #get data file path 
    p = Path(__file__).resolve().parents[2]
    datapath = str(p) + "/preprocessed/input_dataset_for_ML.csv"
    
    #read file
    data = pd.read_csv(datapath)
    data.info()
    
    # x = demand UER trace, y = baseline UER trace
    X = data['dem_uer_trace'].map(lambda x:  unfold_str_v1(x))
    Y = data['base_uer_trace'].map(lambda x:  unfold_str_v1(x))
    
    # create empty list to store values
    diff_dtw = []
    
    # loop to calculate DTW distance
    for idx, val in data.iterrows():
        
        # set X & Y
        base_trace = X[idx]  
        dem_trace = Y[idx]
        X_size = len(base_trace)
        Y_size = len(dem_trace)
        
        # calculate distance
        dtw_val = dtw(base_trace, dem_trace, dist = 'square',
                                        method = 'fast', return_path = False)
        diff_dtw.append(dtw_val)
        
    print("process completed ✓")
    
    # save values of DTW distance into a file
    np.savetxt('dtw_diff.txt', diff_dtw)
    
    