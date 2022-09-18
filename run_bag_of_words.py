'''
Candidate Number: 29441
This code runs the Bag-of-Patterns algorithm.
Majority of this code is written by the candidate,
referencing the 'pyts' package for running the 
Bag-of-Patterns model at 
https://pyts.readthedocs.io/en/stable/generated/pyts.transformation.BagOfPatterns.html
'''

import numpy as np
import matplotlib.pyplot as plt
from pyts.transformation import BagOfPatterns
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from pathlib import Path

# function that extracts each datapoint in each sample
def unfold_str_v1(x): 
    y = x.replace("[","").replace("]","").split(',')
    return [float(y[i]) for i in range(len(y))]

if __name__ == '__main__':
    
    #two classes c_ok and c_dis
    nb_classes = 2
    
    #get data file path 
    p = Path(__file__).resolve().parents[2]
    datapath = str(p) + "/preprocessed/input_dataset_simple.csv"
    
    #read file

    data = pd.read_csv(datapath)
    data.info()
    
    # load data, x = normalised UER trace difference
    X = data['norm_base_dem_diff'].map(lambda x:  unfold_str_v1(x))
    y = data['copper_outcome_mapping_binary'] # 1: c_ok, 0: c_dis
    
    # split data into testing and training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    # transform data to array
    X_train.to_frame
    X_train = pd.DataFrame(X_train.tolist())
    X_train = np.array(X_train)
    y_train = np.array(y_train)

     # create bag of patterns algorithm with a window size of 50,
     # word size of 3
    bop = BagOfPatterns(window_size = 50, word_size = 3, n_bins = 2,
                        numerosity_reduction = False, sparse = False)

    # fit bag of patterns algorithm on training data
    X_bop = bop.fit_transform(X_train)

    # Visualize the bag of patterns algorithm on the data
    plt.figure(figsize=(6, 4))
    vocabulary_length = len(bop.vocabulary_)
    width = 0.3
    plt.bar(np.arange(vocabulary_length) - width / 2, X_bop[y_train == 0][0],
            width=width, label='First time series in class 1')
    plt.bar(np.arange(vocabulary_length) + width / 2, X_bop[y_train == 1][0],
            width=width, label='First time series in class 2')
    plt.xticks(np.arange(vocabulary_length),
               np.vectorize(bop.vocabulary_.get)(np.arange(X_bop[0].size)),
               fontsize=12)
    y_max = np.max(np.concatenate([X_bop[y_train == 0][0],
                                   X_bop[y_train == 1][0]]))
    plt.xlabel("Words", fontsize=14)
    plt.ylabel("Frequencies", fontsize=14)
    plt.title("Bag-of-patterns transformation", fontsize=16)
    plt.legend(loc='best', fontsize=10)
    plt.ylim((0, y_max))
    plt.tight_layout()
    plt.show()