import numpy as np

np.random.seed(123)

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

# Custom activation function
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
from keras import backend as K

from hyperas import optim
from hyperopt.hp import choice
from hyperopt import Trials, STATUS_OK, tpe

class Logus(Activation):
    def __init__(self, activation, **kwargs):
        super(Logus, self).__init__(activation, **kwargs)
        self.__name__ = 'Logus'

def logus(x):
    return (K.log(x*x + 0.1))

get_custom_objects().update({'Logus': Logus(logus)})

#np.random.seed(123)


def data():
    TRAIN = './data/train.tsv'
    TEST  = './data/test.tsv'
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
    return X_train, Y_train, X_test, Y_test



def create_model(X_train, Y_train, X_test, Y_test):
    # define a dense input model
    model = Sequential()
    
    model.add(Dense({{choice([16,32,64,128])}}, input_dim = X_train.shape[1]))
    model.add(Activation({{choice(['Logus','sigmoid','tanh'])}}))
    
    model.add(Dense({{choice([16,32,64,128])}}))
    model.add(Activation({{choice(['Logus','softmax','tanh'])}}))
    model.add(Dropout({{uniform(0,1)}}))
    
    model.add(Dense({{choice([16,32,64,128])}}))
    model.add(Activation({{choice(['Logus','softmax','tanh'])}}))
    model.add(Dropout({{uniform(0,1)}}))
    
    if {{choice(['three', 'four'])}} == 'four':
        model.add(Dense({{choice([16,32,64,128])}}))
        model.add(Activation({{choice(['Logus','softmax','tanh'])}}))
        model.add(Dropout({{uniform(0,1)}}))

    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.compile(
            loss='categorical_crossentropy', 
            metrics=['accuracy'],
            optimizer='adam'
    )

    with tf.device('/cpu:0'):
        checkpoint = ModelCheckpoint('models/model-valacc-{val_acc:03f}.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='auto') 
        # fit model
        result = model.fit(
                X_train, 
                Y_train, 
                batch_size={{choice([8,16,32])}},
                validation_data=(X_test, Y_test), 
                nb_epoch=100, 
                verbose=0, 
                callbacks=[checkpoint]
        )
        validation_acc = np.amax(result.history['val_acc'])
        print('Best validation acc of epoch:', validation_acc)
        return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}
        
        #_, train_acc = saved_model.evaluate(X_train, Y_train, verbose=0)
        #_, test_acc = saved_model.evaluate(X_test, Y_test, verbose=0)
        #print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=100,
                                          trials=Trials())
    X_train, Y_train, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)

sys.exit()
