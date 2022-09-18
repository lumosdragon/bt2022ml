'''
Candidate Number: 29441
This code runs the code for plotting DTW distances
generated from the previous code.
This code is written by the candidate,
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import string

from os import path

if __name__ == '__main__':
    
    #two classes c_ok and c_dis
    nb_classes = 2
    
    #get data file path 
    p = Path(__file__).resolve().parents[2]
    datapath = str(p) + "/preprocessed/input_dataset_for_ML.csv"
    
    #read file
    data = pd.read_csv(datapath)
    
    # load data for binary classes
    y = data['copper_outcome_mapping_binary'] # 1: c_ok, 0: c_dis
    
    # get data file for dtw distance
    datapath = str(p) + '/python files/selt_isabelle/dtw_diff.txt'
    txt_file = open(datapath, 'r')
    x = txt_file.readlines()
    
    # pandas to array
    x = np.array(x, dtype=np.float32)
    y = np.array(y)


    # create dtw distance with corresponding binary outcome
    df = pd.DataFrame({'dtw_dist': x, 'bin_outcome': y}, columns=['dtw_dist', 'bin_outcome'])
    
    # split dtw distance into two classes
    df1 = df[df['bin_outcome'] == 0]
    df2 = df[df['bin_outcome'] == 1]

    # print description of DTW distance data for each class
    print(df1['dtw_dist'].describe())
    print(df2['dtw_dist'].describe())
    
    # create distribution graphs for dtw distance for each class
    fig, (ax1, ax2) = plt.subplots(2, sharex = True)
    fig.suptitle(t='Distribution of DTW Distance by Class')
    ax1.hist(df1['dtw_dist'], log = True, color = 'g', label = 'Class NOK')
    ax1.legend(loc="upper right")
    ax2.hist(df2['dtw_dist'], log = True, color = 'r', label = 'Class OK')
    ax2.legend(loc="upper right")
    
    plt.show()
    
    
    



    
 