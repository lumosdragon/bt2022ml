'''
Candidate Number: 29441
This code is for using CNN on the UER trace differences.
The majority of this code is from BT's Applied Research team
sources from a third-party website.
Parts of the code written by the candidate is commented 
explicitly.
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, auc
import pandas as pd
import seaborn as sns
import scipy.stats as st
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from pathlib import Path
from sklearn.model_selection import train_test_split
from keras.layers import LSTM
from tensorflow.keras import layers
from keras.layers import Conv1D, Conv2D, Flatten, BatchNormalization
from sklearn.utils import class_weight
from keras.layers.pooling import MaxPooling1D
from keras.metrics import categorical_accuracy, AUC
from sklearn.preprocessing import LabelEncoder
import random


def unfold_str_v1(x): 
    y = x.replace("[","").replace("]","").split(',')
    return [float(y[i]) for i in range(len(y))]
    
def indexes_with_correct_outcome(a, b):
  return [i for i, v in enumerate(a) if (v == b[i]).any]
 
def indexes_with_incorrect_outcome(a, b):
  return [i for i, v in enumerate(a) if (v != b[i]).any] 

#----- normalisation methods -----#
#----- written by candidate -----#

def res_norm(arr_x): #residual normalisation
    res_arr = arr_x - np.mean(arr_x)
    return res_arr
    
def norm_by_row(arr_x): #local normalisation by each sample
    row_sum = np.ndarray.sum(arr_x, axis = 1)
    x_norm = arr_x/row_sum[:, None]
    return x_norm

def tanh_est(arr_x): #tanh estimator normalisation
    mean = np.mean(arr_x, axis = 1)
    std = np.std(arr_x, axis = 1)
    x_diff = arr_x - mean[:, None]
    x_div = x_diff / std[:, None]
    x_est = 0.5 * (np.tanh(0.01 * (x_diff / x_div)) + 1)
    return x_est

def softmax_norm(arr_x): #softmax normalisation
    max = np.max(arr_x, axis = 1, keepdims = True)
    e_x = np.exp(arr_x - max)
    sum = np.sum(e_x, axis = 1, keepdims = True)
    f_x = e_x / sum
    return f_x

#------------------------------------------#


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
    datapath = str(p) + "/preprocessed/input_dataset_for_ML.csv"
    
    #read file
    data = pd.read_csv(datapath)
    
    X = data['norm_base_dem_diff'].map(lambda x:  unfold_str_v1(x))
    y = data['copper_outcome_mapping_binary'] # 1: c_ok, 0: c_dis
    
    #dataframe to ndarray
    arr_X = np.zeros((X.size, len(X[0])))
    for idx,val in enumerate(X):
        arr_X[idx,:] = X[idx]
    
    #dataframe to ndarray 
    arr_y = np.zeros((y.size, 1))
    for idx,val in enumerate(y):
        arr_y[idx,:] = y[idx]
        

    #----- normalization -----#
    #----- written by candidate -----#
    
    # use defined functions above to normalise the UER traces
    # as input
    
    X_tanh = tanh_est(arr_X)
    X_softmax = softmax_norm(arr_X)
    X_norm_res = res_norm(arr_X)
    X_row_sum = norm_by_row(arr_X)
    
    #------------------------------------------#
    
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
    
    #----- oversampling and undersampling ------#
    #----- written by candidate -----#
    
    y_train_ok_idx = [i for i, v in enumerate (y_train) if v == 1] # extract indexes for 'ok' samples
    y_train_nok_idx = [i for i, v in enumerate(y_train) if v == 0] # extract indexes for 'nok' samples
    
    # oversampling
    y_train_ok_idx = np.random.choice(y_train_ok_idx, size = 3000, replace = False) #sample 3000 instances randomly
    y_train_nok_idx = np.random.choice(y_train_nok_idx, size =3000, replace = True) #sample 3000 instances randomly
    # undersampling
    y_train_ok_idx = np.random.choice(y_train_ok_idx, size = 1500, replace = False) #sample 1500 instances randomly
    y_train_nok_idx = np.random.choice(y_train_nok_idx, size =1500, replace = False) #sample 1500 instances randomly
    
    y_train_idx = np.append(y_train_ok_idx, y_train_nok_idx) #create list of index for undersmapled/oversampled training data
    y_train_idx.sort()
    y_ttrain = [y_train[i] for i in y_train_idx] #create data for oversampled/undersampled target feature based on selected index
    y_train = np.array(y_ttrain)
    
    X_ttrain = [X_train[i] for i in y_train_idx] #create data for oversampled/undersampled features based on selected index
    X_train = np.array(X_ttrain)
    
    #------------------------------------------#
    
    #inspect y_test and y_train

    plt.hist(y_train)
    plt.title("y_train")
    
    plt.figure()
    plt.hist(y_test)
    plt.title("y_test")
    plt.show()
    
        
    y_train_ok_idx = [i for i, v in enumerate (y_train) if v == 1]
    y_train_nok_idx = [i for i, v in enumerate(y_train) if v == 0]
    y_train_ok_idx = random.sample(y_train_ok_idx, 4)
    y_train_nok_idx = random.sample(y_train_nok_idx, 5)
    
    y_train_ok_idx.sort()
    y_train_nok_idx.sort()
    
    x_ok = [X_train[i] for i in y_train_ok_idx]
    x_nok = [X_train[i] for i in y_train_nok_idx]
    
    
    
    #sample of training data
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.plot(x_ok[i])
    plt.suptitle('Traces of Class OK')
    plt.show()
    
    
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.plot(x_nok[i])
    plt.suptitle('Traces of Class NOK')
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
    
    
    #----- calculate class weights -----#
    #----- written by candidate -----#
    
    c_weights = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(y_train), y = y_train[:,0]) 
    print(c_weights)

    c_weights = dict(enumerate(c_weights))
    print(c_weights)
    
    #------------------------------------------#
    
    
    print("X_train shape", X_train.shape)
    print("X_test shape", X_test.shape)
    print("Y_train shape", Y_train.shape)
    print("Y_test shape", Y_test.shape)
    
    
    
    
    model = Sequential()

    # create and define layers in neural network
    #----- written by candidate -----#
    
    # first layer
    model.add(Conv1D(16, kernel_size=2, activation='relu', input_shape=(4096,1)))
    #model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size = 2, strides = 2, padding = 'same'))
    
    # second layer
    model.add(Conv1D(128, kernel_size=2, activation='relu', input_shape=(4096,1)))
    #model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size = 2, strides = 2, padding = 'same'))
    #model.add(Dropout(0.2))
    #model.add(Conv1D(16, kernel_size=2, activation='relu'))
    
    # fully connected layer
    model.add(Flatten())
    #model.add(Dense(8, activation='softmax'))
    model.add(Dense(2, activation='softmax'))
    
    #model.add(MaxPooling1D(pool_size=8))
    
    # plot shape of model
    plot_model(model, to_file = 'uer_baseline_model.png', show_shapes = True, show_layer_names = True)
    
    #------------------------------------------#
    
    model.build(input_shape=(4096,1))
    print(model.summary())
    


    
    #========================================================================
    #compile the model
    #========================================================================
    
    #loss function we'll use here is called categorical crossentropy, 
    #and is a loss function well-suited to comparing two probability distributions.
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['AUC', 'Accuracy'                             
                                                                                  ])
                  

    #========================================================================
    #train the model
    #========================================================================
    
    #Fitting data to the model
    history = model.fit(X_train, Y_train, 
              batch_size=64, epochs=10, 
              verbose=1, 
              validation_split=0.2)
    
    # fitting data with class weights, written by the candidate
    history = model.fit(X_train, Y_train, 
              batch_size=64, epochs=10, 
              verbose=1, 
              validation_split=0.2,  class_weight=(c_weights))
    
    
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
    plt.plot(history.history['auc'], marker="o")
    plt.plot(history.history['val_auc'], marker="d")
    plt.title('model AUC')
    plt.ylabel('AUC')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    
    plt.figure()
    plt.plot(history.history['Accuracy'], marker="o")
    plt.plot(history.history['val_Accuracy'], marker="d")
    plt.title('model accuracy')
    plt.ylabel('Accuracy')
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
    
    #========================================================================
    #Inspecting the output & Confusion Matrix
    #========================================================================
    
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
    print("Sensitivity/Recall/TP rate ", TPR[0])
    # Specificity or true negative rate
    TNR = TN/(TN+FP)
    print("Specificity/TN rate ", TNR[0])
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    print("Precision/positive predictive rate ", PPV[0])
    # Negative predictive value
    NPV = TN/(TN+FN)
    print("Negative Predictive Value ", NPV[0])
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    print("Fall out/false positive rate ", FPR[0])
    # False negative rate
    FNR = FN/(TP+FN)
    print("False Negative Score ", FNR[0])
    # False discovery rate
    FDR = FP/(TP+FP)
    print("False discovery rate ", FDR[0])
    
    
    #----- additional performance scores and charts -----#
    # written by candidate #
    # Kappa statistic (Agreement rate)
    kappa = ((2 * (TP*TN-FN*FP)) / ((TP+FP) * (FP+TN) + (TP + FN) * (FN + TN))) 
    print('Kappa statistic ', kappa[0])
    # F-1 score
    f1 = ((TP) / ((TP) + (0.5 * (FP + FN))))
    print('F1 Score ', f1[0])
    
    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)
    print("Accuracy", ACC[0])
    
    plt.show()
    
    # AUC-ROC
    fpr, tpr, thresholds = roc_curve(y_test, ypred)
    auc_keras = auc(fpr, tpr)    
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='Keras (area = {:.3f})'.format(auc_keras))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()
    #------------------------------------------#