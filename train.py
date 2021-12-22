import numpy as np

np.random.seed(123)

import matplotlib.pyplot as plt
import keras
from hyperas.distributions import uniform
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
np.random.seed(123)
from keras.models import load_model
from keras.models import Sequential
from keras.optimizers import SGD;
from keras.layers import Convolution1D, Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import pandas as pd
import os
import sys
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Custom activation function
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
from keras import backend as K

class Logus(Activation):
    def __init__(self, activation, **kwargs):
        super(Logus, self).__init__(activation, **kwargs)
        self.__name__ = 'Logus'

def logus(x):
    return (K.log(x*x + 0.1))

get_custom_objects().update({'Logus': Logus(logus)})

#np.random.seed(123)

TRAIN      = './data/train.tsv'
TEST       = './data/test.tsv'

def read_tsv(filename):
    df  = pd.read_csv(filename, header=None, delimiter='\t', names=['id','name','label','spectrum'])
    df = df.sample(frac=1,random_state=12345)
    X = []
    Y = []
    for index, row in df.iterrows(): 
        Y.append( row['label'] == 'cancer' )
        X.append( np.array( row['spectrum'].split(',') ).astype(np.float) )

    X = np.array(X)
    Y = np_utils.to_categorical(np.array(Y), 2).astype(np.int) ## BINARY! 
    return X,Y

X_train, Y_train = read_tsv(TRAIN)
X_test,  Y_test  = read_tsv(TEST)

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

# define a dense input model
model = Sequential()
model.add(Dense(32, input_dim = X_train.shape[1]))
model.add(Activation('sigmoid'))

model.add(Dense(32))
model.add(Activation('tanh'))

model.add(Dropout(0.87))

model.add(Dense(64))
model.add(Activation('tanh'))

model.add(Dropout(0.4))

model.add(Dense(2))
model.add(Activation('softmax'))
# simple early stopping
optimizer = keras.optimizers.Adam(lr=1e-5)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

model.summary()

np.random.seed(123)
with tf.device('/cpu:0'):
    
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5000)
    mc = ModelCheckpoint('model-valacc-{val_acc:03f}.h5', verbose=0, monitor='val_loss',save_best_only=True, mode='auto') 
    
    # fit model
    history = model.fit(
            X_train, 
            Y_train, 
            validation_data=(X_test, Y_test), 
            nb_epoch=15000, 
            batch_size=32,
            verbose=1, 
            callbacks=[es, mc]
    )
    np.save('./history.npy',history.history)
    # history=np.load('my_history.npy',allow_pickle='TRUE').item()


    if False:
        plt.grid()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
    # load the saved model
    #evaluate the model
    #_, train_acc = saved_model.evaluate(X_train, Y_train, verbose=0)
    #_, test_acc = saved_model.evaluate(X_test, Y_test, verbose=0)
    #print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

sys.exit()
