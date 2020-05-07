from __future__ import absolute_import
from __future__ import print_function

import os
import multiprocessing as mp
from subprocess import call
import warnings
import numpy as np
import scipy.io as sio
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import scale
import keras.backend as K
from keras.datasets import mnist, cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.regularizers import l2
import tensorflow as tf
from scipy.spatial.distance import pdist, cdist, squareform
from keras import regularizers
from sklearn.decomposition import PCA
from keras.layers import BatchNormalization



# CLIP_MIN = 0.0
# CLIP_MAX = 1.0
#CLIP_MIN = -0.5
#CLIP_MAX = 0.5
PATH_DATA = "data/svhn/"

# Set random seed
np.random.seed(0)


def get_data(dataset='mnist'):
    """
    images in [-0.5, 0.5] (instead of [0, 1]) which suits C&W attack and generally gives better performance
    
    :param dataset:
    :return: 
    """
    assert dataset in ['mnist', 'cifar', 'svhn'], \
        "dataset parameter must be either 'mnist' 'cifar' or 'svhn'"
    if dataset == 'mnist':
        # the data, shuffled and split between train and test sets
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        # reshape to (n_samples, 28, 28, 1)
        X_train = X_train.reshape(-1, 28, 28, 1)
        X_test = X_test.reshape(-1, 28, 28, 1)
    elif dataset == 'cifar':
        # the data, shuffled and split between train and test sets
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    else:
        if not os.path.isfile(os.path.join(PATH_DATA, "svhn_train.mat")):
            print('Downloading SVHN train set...')
            call(
                "curl -o ../data/svhn_train.mat "
                "http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                shell=True
            )
        if not os.path.isfile(os.path.join(PATH_DATA, "svhn_test.mat")):
            print('Downloading SVHN test set...')
            call(
                "curl -o ../data/svhn_test.mat "
                "http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
                shell=True
            )
        train = sio.loadmat(os.path.join(PATH_DATA,'svhn_train.mat'))
        test = sio.loadmat(os.path.join(PATH_DATA, 'svhn_test.mat'))
        X_train = np.transpose(train['X'], axes=[3, 0, 1, 2])
        #X_test = np.transpose(test['X'], axes=[3, 0, 1, 2])
        X_test = test['X'].transpose(3, 0, 1, 2)
        # reshape (n_samples, 1) to (n_samples,) and change 1-index
        # to 0-index
        y_train = np.reshape(train['y'], (-1,)) - 1
        #y_test = np.reshape(test['y'], (-1,)) - 1
        y_test = test['y']
        y_test[y_test==10] = 0

    # cast pixels to floats, normalize to [0, 1] range
    #X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    
    #X_train = (X_train/255.0)
    X_test = (X_test/255.0) 
    #X_train = (X_train/255.0) - (1.0 - CLIP_MAX)
    #X_test = (X_test/255.0) - (1.0 - CLIP_MAX)

    # one-hot-encode the labels
    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)

    #print("X_train:", X_train.shape)
    #print("Y_train:", Y_train.shape)
    print("X_test:", X_test.shape)
    print("Y_test", Y_test.shape)

    return X_train, Y_train, X_test, Y_test

def get_model(dataset='mnist', softmax=True):
    """
    Takes in a parameter indicating which model type to use ('mnist',
    'cifar' or 'svhn') and returns the appropriate Keras model.
    :param dataset: A string indicating which dataset we are building
                    a model for.
    :param softmax: if add softmax to the last layer.
    :return: The model; a Keras 'Sequential' instance.
    """
    assert dataset in ['mnist', 'cifar', 'svhn'], \
        "dataset parameter must be either 'mnist' 'cifar' or 'svhn'"
    if dataset == 'mnist':
        # MNIST model: 0, 2, 7, 10
        layers = [
            Conv2D(64, (3, 3), padding='valid', input_shape=(28, 28, 1)),  # 0
            Activation('relu'),  # 1
            #BatchNormalization(), # 2
            Conv2D(64, (3, 3)),  # 3
            Activation('relu'),  # 4
            #BatchNormalization(), # 5
            MaxPooling2D(pool_size=(2, 2)),  # 6
            Dropout(0.5),  # 7
            Flatten(),  # 8
            Dense(128),  # 9            
            Activation('relu'),  # 10
            #BatchNormalization(), # 11
            Dropout(0.5),  # 12
            Dense(10),  # 13
        ]
    elif dataset == 'cifar':
        # CIFAR-10 model
        layers = [
            Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3)),  # 0
            Activation('relu'),  # 1
            #BatchNormalization(), # 2
            Conv2D(32, (3, 3), padding='same'),  # 3
            Activation('relu'),  # 4
            #BatchNormalization(), # 5
            MaxPooling2D(pool_size=(2, 2)),  # 6
            
            Conv2D(64, (3, 3), padding='same'),  # 7
            Activation('relu'),  # 8
            #BatchNormalization(), # 9
            Conv2D(64, (3, 3), padding='same'),  # 10
            Activation('relu'),  # 11
            #BatchNormalization(), # 12
            MaxPooling2D(pool_size=(2, 2)),  # 13
            
            Conv2D(128, (3, 3), padding='same'),  # 14
            Activation('relu'),  # 15
            #BatchNormalization(), # 16
            Conv2D(128, (3, 3), padding='same'),  # 17
            Activation('relu'),  # 18
            #BatchNormalization(), # 19
            MaxPooling2D(pool_size=(2, 2)),  # 20
            
            Flatten(),  # 21
            Dropout(0.5),  # 22
            
            Dense(1024, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),  # 23
            Activation('relu'),  # 24
            #BatchNormalization(), # 25
            Dropout(0.5),  # 26
            Dense(512, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),  # 27
            Activation('relu'),  # 28
            #BatchNormalization(), # 29
            Dropout(0.5),  # 30
            Dense(10),  # 31
        ]
    else:
        # SVHN model
        layers = [
            Conv2D(64, (3, 3), padding='valid', input_shape=(32, 32, 3)),  # 0
            Activation('relu'),  # 1
            BatchNormalization(), # 2
            Conv2D(64, (3, 3)),  # 3
            Activation('relu'),  # 4
            BatchNormalization(), # 5
            MaxPooling2D(pool_size=(2, 2)),  # 6
            
            Dropout(0.5),  # 7
            Flatten(),  # 8
            
            Dense(512),  # 9
            Activation('relu'),  # 10
            BatchNormalization(), # 11
            Dropout(0.5),  # 12
            
            Dense(128),  # 13
            Activation('relu'),  # 14
            BatchNormalization(), # 15
            Dropout(0.5),  # 16
            Dense(10),  # 17
        ]

    model = Sequential()
    for layer in layers:
        model.add(layer)
    if softmax:
        model.add(Activation('softmax'))

    return model

def cross_entropy(y_true, y_pred):
    return tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)

def lid_term(logits, batch_size=100):
    """Calculate LID loss term for a minibatch of logits

    :param logits: 
    :return: 
    """
    # y_pred = tf.nn.softmax(logits)
    y_pred = logits

    # calculate pairwise distance
    r = tf.reduce_sum(tf.square(y_pred), axis=1)
    # turn r into column vector
    r = tf.reshape(r, [-1, 1])
    D = r - 2 * tf.matmul(y_pred, tf.transpose(y_pred)) + tf.transpose(r)

    # find the k nearest neighbor
    D1 = tf.sqrt(D + 1e-9)
    D2, _ = tf.nn.top_k(-D1, k=21, sorted=True)
    D3 = -D2[:, 1:]

    m = tf.transpose(tf.multiply(tf.transpose(D3), 1.0 / D3[:, -1]))
    v_log = tf.reduce_sum(tf.log(m + 1e-9), axis=1)  # to avoid nan
    lids = -20 / v_log

    ## batch normalize lids
    # lids = tf.nn.l2_normalize(lids, dim=0, epsilon=1e-12)

    return lids

def lid_adv_term(clean_logits, adv_logits, batch_size=100):
    """Calculate LID loss term for a minibatch of advs logits

    :param logits: clean logits
    :param A_logits: adversarial logits
    :return: 
    """
    # y_pred = tf.nn.softmax(logits)
    c_pred = tf.reshape(clean_logits, (batch_size, -1))
    a_pred = tf.reshape(adv_logits, (batch_size, -1))

    # calculate pairwise distance
    r_a = tf.reduce_sum(tf.square(a_pred), axis=1)
    # turn r_a into column vector
    r_a = tf.reshape(r_a, [-1, 1])

    r_c = tf.reduce_sum(tf.square(c_pred), axis=1)
    # turn r_c into row vector
    r_c = tf.reshape(r_c, [1, -1])

    D = r_a - 2 * tf.matmul(a_pred, tf.transpose(c_pred)) + r_c

    # find the k nearest neighbor
    D1 = tf.sqrt(D + 1e-9)
    D2, _ = tf.nn.top_k(-D1, k=21, sorted=True)
    D3 = -D2[:, 1:]

    m = tf.transpose(tf.multiply(tf.transpose(D3), 1.0 / D3[:, -1]))
    v_log = tf.reduce_sum(tf.log(m + 1e-9), axis=1)  # to avoid nan
    lids = -20 / v_log

    ## batch normalize lids
    lids = tf.nn.l2_normalize(lids, dim=0, epsilon=1e-12)

    return lids


def get_layer_wise_activations(model, dataset):
    """
    Get the deep activation outputs.
    :param model:
    :param dataset: 'mnist', 'cifar', 'svhn', has different submanifolds architectures  
    :return: 
    """
    assert dataset in ['mnist', 'cifar', 'svhn'], \
        "dataset parameter must be either 'mnist' 'cifar' or 'svhn'"
    if dataset == 'mnist':
        # mnist model
        acts = [model.layers[0].input]
        acts.extend([layer.output for layer in model.layers])
    elif dataset == 'cifar':
        # cifar-10 model
        acts = [model.layers[0].input]
        acts.extend([layer.output for layer in model.layers])
    else:
        # svhn model
        acts = [model.layers[0].input]
        acts.extend([layer.output for layer in model.layers])
    return acts

# lid of a single query point x
def mle_single(data, x, k=20):
    data = np.asarray(data, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    # print('x.ndim',x.ndim)
    if x.ndim == 1:
        x = x.reshape((-1, x.shape[0]))
    # dim = x.shape[1]

    k = min(k, len(data)-1)
    f = lambda v: - k / np.sum(np.log(v/v[-1]))
    a = cdist(x, data)
    a = np.apply_along_axis(np.sort, axis=1, arr=a)[:,1:k+1]
    a = np.apply_along_axis(f, axis=1, arr=a)
    return a[0]

# lid of a batch of query points X
def mle_batch(data, batch, k):
    data = np.asarray(data, dtype=np.float32)
    batch = np.asarray(batch, dtype=np.float32)

    k = min(k, len(data)-1)
    f = lambda v: - k / np.sum(np.log(v/v[-1]))
    a = cdist(batch, data)
    a = np.apply_along_axis(np.sort, axis=1, arr=a)[:,1:k+1]
    a = np.apply_along_axis(f, axis=1, arr=a)
    return a

# mean distance of x to its k nearest neighbours
def kmean_batch(data, batch, k):
    data = np.asarray(data, dtype=np.float32)
    batch = np.asarray(batch, dtype=np.float32)

    k = min(k, len(data)-1)
    f = lambda v: np.mean(v)
    a = cdist(batch, data)
    a = np.apply_along_axis(np.sort, axis=1, arr=a)[:,1:k+1]
    a = np.apply_along_axis(f, axis=1, arr=a)
    return a

# mean distance of x to its k nearest neighbours
def kmean_pca_batch(data, batch, k=10):
    data = np.asarray(data, dtype=np.float32)
    batch = np.asarray(batch, dtype=np.float32)
    a = np.zeros(batch.shape[0])
    for i in np.arange(batch.shape[0]):
        tmp = np.concatenate((data, [batch[i]]))
        tmp_pca = PCA(n_components=2).fit_transform(tmp)
        a[i] = kmean_batch(tmp_pca[:-1], tmp_pca[-1], k=k)
    return a



def normalize(normal, adv, noisy):
    """Z-score normalisation
    TODO
    :param normal:
    :param adv:
    :param noisy:
    :return:
    """
    n_samples = len(normal)
    total = scale(np.concatenate((normal, adv, noisy)))

    return total[:n_samples], total[n_samples:2*n_samples], total[2*n_samples:]


def train_lr(X, y):
    """
    TODO
    :param X: the data samples
    :param y: the labels
    :return:
    """
    lr = LogisticRegressionCV(n_jobs=-1).fit(X, y)
    return lr

