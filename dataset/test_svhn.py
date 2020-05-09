#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 07:07:13 2020

@author: qq
"""
import scipy.io as sio
import RCNN
def getData(filename):
    load_data = sio.loadmat(filename)
    y = load_data['y']
    X = load_data['X'].transpose(3, 0, 1, 2)
    return X, y

file = 'data/svhn/svhn_test.mat'
X_raw, y_raw = getData(filename=file)
X_raw = X_raw.astype('float32')
n_test = X_raw.shape[0]
y_raw[y_raw==10] = 0

from keras.layers import *
from keras.utils import np_utils


from keras.models import load_model
#model = load_model('model-RCNN_new.hdf5')
model = RCNN.get_model(True)
model_file="../model/model-svhn.h5"
model.load_weights(model_file)

y_raw = np_utils.to_categorical(y_raw, 10)
score = model.evaluate(X_raw, y_raw,verbose=0)
print(score[1])

from util import get_data
_, _, X_test, Y_test = get_data('svhn')
score = model.evaluate(X_test, Y_test,verbose=0)
print(score[1])