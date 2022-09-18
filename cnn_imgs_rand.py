#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 17:22:59 2022

@author: tingys
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 16:59:38 2022

@author: tingys
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import keras
import random
import seaborn as sns
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from pathlib import Path
from sklearn.model_selection import train_test_split
from keras.layers import LSTM, Bidirectional
from tensorflow.keras import layers
from keras.layers import Conv1D, Conv2D, Flatten, BatchNormalization
from keras.layers.pooling import MaxPooling1D

import glob
from PIL import Image


plt.rcParams['figure.figsize'] = (7,7) # Make the figures a bit bigger

def unfold_str_v1(x): 
    y = x.replace("[","").replace("]","").split(',')
    return [float(y[i]) for i in range(len(y))]
    
def indexes_with_correct_outcome(a, b):
  return [i for i, v in enumerate(a) if (v == b[i]).any]
 
def indexes_with_incorrect_outcome(a, b):
  return [i for i, v in enumerate(a) if (v != b[i]).any] 


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

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
    
    X = data['norm_base_dem_diff'].map(lambda x:  unfold_str_v1(x))
    y = data['copper_outcome_mapping_binary'] # 1: c_ok, 0: c_dis
    
    
    print("create images")
    
    #X = X.map(lambda x: GAFS(x)) 
    
    X_temp = X
    show_img = False
    #for filename in glob.glob(str(p)+ "/data/imgs/gafs/*.png"):
    #for filename in glob.glob(str(p)+ "/data/imgs/gafd/*.png"):
    for filename in glob.glob(str(p)+ "/imgs/gafs/*.png"):
        #s = filename
        read_img = Image.open(filename)
        read_img_to_array = np.asarray(read_img)
        
        filename = str(filename)
        filename = filename.replace("_"," ")
        filename = filename.replace(".png"," str_end")
        get_img_idx = [int(s) for s in filename.split() if s.isdigit()][0]
        
        
        if(show_img):
            plt.figure()
            plt.imshow(read_img_to_array)
            plt.show()
            plt.close()
        
        #convert rgb to gray 
        gray = rgb2gray(read_img_to_array)   
        #normalise gray img
        gray_norm = gray/255.0
        
        if(show_img):
            plt.figure()
            #plt.imshow(gray)
            plt.imshow(gray_norm)
            plt.show()
            plt.close()
        
        #reshape image to 1d
        reshape_img_1d = gray_norm.reshape(gray_norm.shape[0]*gray_norm.shape[1],1)
        
        X_temp[get_img_idx] = reshape_img_1d
    
    print("===")
    print(X.shape)
    print(X.size)
    print(len(X[0])) 
    
    X = X_temp
    
    print("___")
    print(X.shape)
    print(X.size)
    print(len(X[0])) 
    

    
    print("create images completed ✓")
    
    print(X.size) #8112
    print(len(X[0])) #136161
    
    #dataframe to ndarray
    
    arr_X = np.zeros((X.size, len(X[0])))
    print("len(arr_X[0,:]) = ", len(arr_X[0,:]))
    print("X[idx] = ", len(X[0]))
    for idx,val in enumerate(X):
        arr_X[idx,:] = X[idx].ravel()
        
    #dataframe to ndarray 
    arr_y = np.zeros((y.size, 1))
    for idx,val in enumerate(y):
        arr_y[idx,:] = y[idx]
    
    X = arr_X 
    y = arr_y
    
    print("X shape", X.shape)
    print("y shape", y.shape)
    
    #split data using sklearn
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    
    # the data, shuffled and split between train and test sets
    print("X_train original shape", X_train.shape)
    print("y_train original shape", y_train.shape)
    print("X_test original shape", X_test.shape)
    print("y_test original shape", y_test.shape)
    print("X_train.shape[1:]", X_train.shape[1:])
    
    X_ttrain = X_train
    
    rand_idx = random.sample(range(y_train.shape[0]-1), 3000)
    
    y_train_ok_idx = [i for i, v in enumerate (y_train) if v == 1]
    y_train_nok_idx = [i for i, v in enumerate(y_train) if v == 0]
    y_train_ok_idx = random.sample(y_train_ok_idx, 2500)
    y_train_idx = y_train_ok_idx + y_train_nok_idx
    
    y_train_idx = y_train_ok_idx + y_train_nok_idx
    y_train_idx.sort()
    y_ttrain = [y_train[i] for i in y_train_idx]
    y_train = np.array(y_ttrain)
    
    X_ttrain = [X_train[i] for i in y_train_idx]
    X_train = np.array(X_ttrain)
    
    #inspect y_test and y_train
    
    plt.hist(y_train)
    plt.title("y_train")
    
    plt.figure()
    plt.hist(y_test)
    plt.title("y_test")
    
    
    # 1: c_ok, 0: c_dis
    print("y_train OK", int(np.sum(y_train)))
    print("y_train DIS", int(len(y_train) - np.sum(y_train)) )
    
    print("y_test OK", int(np.sum(y_test)))
    print("y_test DIS", int(len(y_test) - np.sum(y_test)) )
    
    plt.show()
    
    #sample of training data
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.plot(X_train[i])
        plt.title("Class {}".format(y_train[i]))
    
    plt.show()
    
    #========================================================================
    #Format the data for training
    #========================================================================
    
    #modify target to one-hot format e.g.
    #0 -> [1, 0]
    #1 -> [0, 1]
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    
    #binary classification
    #Y_train = y_train
    #Y_test = y_test
    
    print("Y_train shape", Y_train.shape)
    print("Y_test shape", Y_test.shape)
    
    #========================================================================
    #build the neural network 
    #========================================================================
    print("------------------")
    print("X_train shape", X_train.shape)
    print("X_test shape", X_test.shape)
    
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)  # the 1 is the steps
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)  # the 1 is the steps
    
    print("X_train shape", X_train.shape)
    print("X_test shape", X_test.shape)
    print("Y_train shape", Y_train.shape)
    print("Y_test shape", Y_test.shape)
    
    
    model = Sequential()
    
    #add model layers
    model.add(Conv1D(16, 3, strides = 2, padding = 'same', activation='relu', input_shape=(136161,1)))
    #model.add(MaxPooling1D(pool_size = 3, strides = 2, padding = 'same'))
    
    #model.add(Dropout(0.2))
    #model.add(Conv1D(16, kernel_size=2, activation='relu'))
    model.add(Flatten())
    #model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax'))
    
    #model.add(MaxPooling1D(pool_size=8))
    
    
    
    model.build(input_shape=(4096,1))
    print(model.summary())
    
    #========================================================================
    #compile the model
    #========================================================================
    
    #loss function we'll use here is called categorical crossentropy, 
    #and is a loss function well-suited to comparing two probability distributions.
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    
    #Compiling the model
    #model.compile( loss='sparse_categorical_crossentropy', optimizer = keras.optimizers.Adam(lr=0.001, decay=1e-6), metrics=['accuracy'] )
    
    
    #========================================================================
    #train the model
    #========================================================================
    
    history = model.fit(X_train, Y_train, 
              batch_size=64, epochs=10, 
              verbose=1, 
              validation_split=0.2)
    
    #history = model.fit(X_train, Y_train,validation_data=(X_test, y_test), epochs=10)
    
    #Fitting data to the model
    #history = model.fit(X_train, y_train, epochs=3, validation_data=(X_test, Y_test))
    
    print(history)
    
    #========================================================================
    #performance evaluation 
    #========================================================================
    
    score = model.evaluate(X_test, Y_test, verbose=0) 
    
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    
    #========================================================================
    #plot accuracy/loss
    #========================================================================
    
    plt.figure()
    plt.plot(history.history['accuracy'], marker="o")
    plt.plot(history.history['val_accuracy'], marker="d")
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    
    plt.figure()
    plt.plot(history.history['loss'], marker="o")
    plt.plot(history.history['val_loss'], marker="d")
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    
    inspect_output = 0
    
    print("\nInspecting the output")
    
    # The predict_classes function outputs the highest probability class
    # according to the trained classifier for each input example.
    predicted_classes = model.predict(X_test)
    
    # Check which items we got right / wrong
    correct_indices = indexes_with_correct_outcome(predicted_classes,y_test)
    incorrect_indices = indexes_with_incorrect_outcome(predicted_classes,y_test)
    
    testsamplesize= len(correct_indices)+len(incorrect_indices)
    print(len(correct_indices), "correct out of ", testsamplesize, " ~", len(correct_indices)/testsamplesize, "%")
    print(len(incorrect_indices), "incorrect out of ", testsamplesize, " ~", len(incorrect_indices)/testsamplesize, "%")
    
    if(inspect_output):
        plt.figure()
        #for i, correct in enumerate(correct_indices[10:19]):
        for i, correct in enumerate(correct_indices[:9]):
            plt.subplot(3,3,i+1)
            plt.plot(X_test[i])
            plt.title("Predicted {}, Class {}".format(predicted_classes[correct], y_test[correct]))
            
        plt.figure()
        for i, incorrect in enumerate(incorrect_indices[:9]):
            plt.subplot(3,3,i+1)
            plt.plot(X_test[i])
            plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], y_test[incorrect]))
        
        plt.show()
        #plt.close()
    
    #confusion matrix
    
    
    # get predict prob and label 
    ypred = model.predict(X_test, verbose=1)
    ypred = np.argmax(ypred, axis=1)
    
    '''
    # as I've trained my model on MNIST as odd or even (binary classes)
    target_names = ['OK', 'DIS']
    
    
    print(classification_report(np.argmax(Y_test, axis=1), ypred, target_names=target_names))
    '''
    print("Confusion Matrix outputs")
    okok = 0
    okdis = 0
    disok = 0
    disdis = 0
    
    for i in range(len(ypred)):
        if(ypred[i] == 0):
            if(y_test[i] == 0):
                okok+=1
            else:
                okdis+=1
        else:
            if(y_test[i] == 0):
                disok+=1
            else:
                disdis+=1
            
    print("len_y_pred = ", len(ypred))
    print("okok ", okok)
    print("okdis ", okdis)
    print("disok ", disok)
    print("disdis ", disdis)
    
    #cm = confusion_matrix(np.argmax(y_train, axis=1), ypred)
    cm = confusion_matrix(y_test, ypred)
    cm = pd.DataFrame(cm, range(2),range(2))
    #plt.figure(figsize = (10,10))
    ax= plt.subplot()
    
    sns.heatmap(cm, annot=True, annot_kws={"size": 12}, fmt='g') # font size
    
    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['DIS', 'OK']); ax.yaxis.set_ticklabels(['DIS', 'OK'])
    
    FP = cm.sum(axis=0) - np.diag(cm)  
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.values.sum() - (FP + FN + TP)
    
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    print("TP", TPR[0])
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    print("TN", TNR[0])
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    print("Precision", PPV[0])
    # Negative predictive value
    NPV = TN/(TN+FN)
    print("NPV", NPV[0])
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    print("FP", FPR[0])
    # False negative rate
    FNR = FN/(TP+FN)
    print("FN", FNR[0])
    # False discovery rate
    FDR = FP/(TP+FP)
    print("FDR", FDR[0])
    
    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)
    print("Accuracy", ACC[0])
    
    plt.show()
    
