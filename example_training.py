"""
This function trains and evaluates a model from RFML

Requires: Python 3.5
Last modified: 9/13/2017
Last modified by: Boston Clark Terry
"""

import os, random
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["THEANO_FLAGS"]  = "device=gpu%d"%(1)
import numpy as np
from keras.utils import np_utils
import keras.models as models
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.regularizers import *
from keras.optimizers import adam
import matplotlib.pyplot as plt
import pickle
import random, sys, keras
import pdb

# Load the dataset ...
dataFile = "../Prototype/Data/Python2.7/RML2016.10a_dict.dat"
Xd = pickle.load(open(dataFile,'rb'),encoding='latin1') # latin encoding used for python3 support
# Xd - data (as dictionary)

# Extract the labels
snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
# snrs:  [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
# mods: ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']
X = []  
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod,snr)])
        for i in range(Xd[(mod,snr)].shape[0]):  lbl.append((mod,snr))
# stack arrays in sequence vertically
X = np.vstack(X) # len(X) -> 220,000   len(lbl) -> 220,000


# Partition the data into training and test sets of the form we can train/test on  while keeping SNR and Mod labels handy for each
np.random.seed(2016)
n_examples = X.shape[0] # dim(X) -> [220000, 2, 128]
# split the data into half (training, testing)
n_train = int(n_examples * 0.5)
train_idx = np.random.choice(range(0,n_examples), size=n_train, replace=False)
test_idx = list(set(range(0,n_examples))-set(train_idx))
X_train = X[train_idx]
X_test =  X[test_idx]

# map the labels to outputs
def to_onehot(yy):
	yy_list = list(yy)
	yy1 = np.zeros([len(yy_list), max(yy_list)+1]) # np.zeros(2,1) -> array([[0.0], 0.0]])
	yy1[np.arange(len(yy_list)), yy_list] = 1
	return yy1

Y_train = to_onehot(map(lambda x: mods.index(lbl[x][0]), train_idx))
Y_test = to_onehot(map(lambda x: mods.index(lbl[x][0]), test_idx))
	
# set the input shape [2,128]
in_shp = list(X_train.shape[1:])
print (X_train.shape, in_shp) # (110000, 2, 128) [2, 128]
classes = mods


# Build VT-CNN2 Neural Net model using Keras primitives -- 
#  - Reshape [N,2,128] to [N,1,2,128] on input
#  - Pass through 2 2DConv/ReLu layers
#  - Pass through 2 Dense layers (ReLu and Softmax)
#  - Perform categorical cross entropy optimization

dr = 0.5 # dropout rate (%)
model = models.Sequential()
model.add(Reshape([1]+in_shp, input_shape=in_shp))
model.add(ZeroPadding2D((0, 2), data_format="channels_first"))
model.add(Convolution2D(256, (1, 3), activation="relu", name="conv1", data_format="channels_first"))
model.add(Dropout(dr))
model.add(ZeroPadding2D((0, 2), data_format="channels_first"))
model.add(Convolution2D(80, (2, 3), activation="relu", name="conv2", data_format="channels_first"))
model.add(Dropout(dr))
model.add(Flatten())
model.add(Dense(256, activation='relu', kernel_initializer='he_normal', name="dense1"))
model.add(Dropout(dr))
model.add(Dense( len(classes), kernel_initializer='he_normal', name="dense2" ))
model.add(Activation('softmax'))
model.add(Reshape([len(classes)]))
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()


#____________________________________________________________________________________________________
#Layer (type)                     Output Shape          Param #     Connected to                     
#====================================================================================================
#reshape_1 (Reshape)              (None, 1, 2, 128)     0           reshape_input_1[0][0]            
#____________________________________________________________________________________________________
#zeropadding2d_1 (ZeroPadding2D)  (None, 1, 2, 132)     0           reshape_1[0][0]                  
#____________________________________________________________________________________________________
#conv1 (Convolution2D)            (None, 256, 2, 130)   1024        zeropadding2d_1[0][0]            
#____________________________________________________________________________________________________
#dropout_1 (Dropout)              (None, 256, 2, 130)   0           conv1[0][0]                      
#____________________________________________________________________________________________________
#zeropadding2d_2 (ZeroPadding2D)  (None, 256, 2, 134)   0           dropout_1[0][0]                  
#____________________________________________________________________________________________________
#conv2 (Convolution2D)            (None, 80, 1, 132)    122960      zeropadding2d_2[0][0]            
#____________________________________________________________________________________________________
#dropout_2 (Dropout)              (None, 80, 1, 132)    0           conv2[0][0]                      
#____________________________________________________________________________________________________
#flatten_1 (Flatten)              (None, 10560)         0           dropout_2[0][0]                  
#____________________________________________________________________________________________________
#dense1 (Dense)                   (None, 256)           2703616     flatten_1[0][0]                  
#____________________________________________________________________________________________________
#dropout_3 (Dropout)              (None, 256)           0           dense1[0][0]                     
#____________________________________________________________________________________________________
#dense2 (Dense)                   (None, 11)            2827        dropout_3[0][0]                  
#____________________________________________________________________________________________________
#activation_1 (Activation)        (None, 11)            0           dense2[0][0]                     
#____________________________________________________________________________________________________
#reshape_2 (Reshape)              (None, 11)            0           activation_1[0][0]               
#====================================================================================================
#Total params: 2830427


# Set up some params 
nb_epoch = 50     # number of epochs to train on
#batch_size = 1024  # training batch size
# OOM
batch_size = 256

# perform training ...
#   - call the main training loop in keras for our network+dataset
filepath = 'convmodrecnets_CNN2_0.5.wts.h5'
history = model.fit(X_train,
                    Y_train,
                    batch_size=batch_size,
                    epochs=nb_epoch,
                    verbose=2,
                    validation_data=(X_test, Y_test),
                    callbacks = [
        keras.callbacks.ModelCheckpoint(filepath, save_best_only=True),
        keras.callbacks.EarlyStopping(patience=5)
        ])

# we re-load the best weights once training is finished
model.load_weights(filepath)


# Show simple version of performance
score = model.evaluate(X_test, Y_test, verbose=0, batch_size=batch_size)
print(score) # 1.27649493232

# stop for debugging

# Show loss curves 
plt.figure() # creates a new figure
plt.title('Training performance')
plt.plot(history.epoch, history.history['loss'], label='train loss+error') # plot: [x:epoch, y:loss]
plt.plot(history.epoch, history.history['val_loss'], label='val_error')    # plot: [x:epoch, y:val_error]
plt.legend()

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
 
# Plot confusion matrix
test_Y_hat = model.predict(X_test, batch_size=batch_size) # generates output predictions for the inputs

conf = np.zeros([len(classes),len(classes)]) # conf: [11x11]

confnorm = np.zeros([len(classes),len(classes)]) # confnorm: [11x11]

for i in range(0,X_test.shape[0]):
    j = list(Y_test[i,:]).index(1)
    k = int(np.argmax(test_Y_hat[i,:]))
    conf[j,k] = conf[j,k] + 1 # increment each time

for i in range(0,len(classes)):
    confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])

plot_confusion_matrix(confnorm, labels=classes)  


# These errors are probably from trying to divide by zero or NaN
# example_training.py:208: RuntimeWarning: invalid value encountered in true_divide
#  confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])

# example_training.py:214: RuntimeWarning: invalid value encountered in double_scalars
#  print ("Overall Accuracy: ", cor / (cor+ncor))

# Overall Accuracy:  nan

# example_training.py:215: RuntimeWarning: invalid value encountered in double_scalars
#  acc[snr] = 1.0*cor/(cor+ncor)

# just in case we picked up some NaNs
np.nan_to_num(confnorm)

# Plot confusion matrix
acc = {}
for snr in snrs:

    # extract classes @ SNR
    test_SNRs = list(map(lambda x: lbl[x][1], test_idx))
    test_X_i = X_test[np.where(np.array(test_SNRs)==snr)]
    test_Y_i = Y_test[np.where(np.array(test_SNRs)==snr)]    

    # estimate classes
    test_Y_i_hat = model.predict(test_X_i)
    conf = np.zeros([len(classes),len(classes)])
    confnorm = np.zeros([len(classes),len(classes)])

    for i in range(0,test_X_i.shape[0]):
        j = list(test_Y_i[i,:]).index(1)
        k = int(np.argmax(test_Y_i_hat[i,:]))
        conf[j,k] = conf[j,k] + 1

    for i in range(0,len(classes)):
        confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
    
    np.nan_to_num(confnorm)
    plot_confusion_matrix(confnorm, labels=classes, title="ConvNet Confusion Matrix (SNR=%d)"%(snr))
    
    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
    print ("Overall Accuracy: ", cor / (cor+ncor))
    acc[snr] = 1.0*cor/(cor+ncor)
    
  # Save results to a pickle file for plotting later
print (acc)
fd = open('results_cnn2_d0.5.dat','wb')
pickle.dump( ("CNN2", 0.5, acc) , fd )

  

# Plot accuracy curve
plt.xlabel("Signal to Noise Ratio")
plt.ylabel("Classification Accuracy")
plt.title("CNN2 Classification Accuracy on RadioML 2016.10 Alpha")
plt.show()


