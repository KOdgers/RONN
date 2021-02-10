from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import unittest
import sys
sys.path.append(".")


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import fetch_covtype
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from Submodules.SaladWrap import *

def load_data_covtype():
    print("Loading forest cover dataset...")

    cover = fetch_covtype()
    df = pd.DataFrame(cover['data'], columns=cover['feature_names'])
    target = cover['target']
    target = target - 1
    target = to_categorical(target, 7)

    return train_test_split(np.array(df)[:10000], np.array(target)[:10000], test_size=.3)


def load_fashion():
    print('TensorFlow version: {}'.format(tf.__version__))

    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)
    # scale the values to 0.0 to 1.0
    scale = np.max([np.max(train_images), np.max(test_images)])
    train_images = train_images / scale
    test_images = test_images / scale

    # reshape for feeding into the model
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

    return train_images, test_images, train_labels, test_labels


class MyTestCase(unittest.TestCase):

    def test_model_all(self):
        XT, xt, YT, yt = load_data_covtype()
        # print(XT.shape,xt.shape,YT.shape,yt.shape)
        input = keras.Input(shape=(XT.shape[1]))
        X = keras.layers.Dense(5,activation='relu')(input)
        X = keras.layers.Dense(5,activation='relu')(X)
        X = keras.layers.Dense(10,activation='relu')(X)
        out = keras.layers.Dense(YT.shape[1], activation='softmax')(X)
        lca_model = MultiSGDLCA(inputs=[input], outputs=out, lca_type='Mean',optimizer_allocation='all')
        # lca_model.compile(optimizer='adam',
        #                   loss='categorical_crossentropy',
        #                   metrics=['accuracy'])
        lca_model.optimizer_allocation(lr=.0001)
        lca_model.setup_lca_save(path='', basename='1D', occurence=10)
        lca_model.Fit(x=XT, y=YT, validation_data=(xt, yt), epochs=20,batch_size=64)

    def test_model_edge(self):
        XT, xt, YT, yt = load_data_covtype()
        # print(XT.shape,xt.shape,YT.shape,yt.shape)
        input = keras.Input(shape=(XT.shape[1]))
        X = keras.layers.Dense(5,activation='relu')(input)
        X = keras.layers.Dense(5,activation='relu')(X)
        # X = keras.layers.Dense(5)(X)
        X = keras.layers.Dense(5,activation='relu')(X)
        X = keras.layers.Dense(10,activation='relu')(X)
        out = keras.layers.Dense(YT.shape[1], activation='softmax')(X)
        lca_model = MultiSGDLCA(inputs=[input], outputs=out, lca_type='Mean',optimizer_allocation='edges',layer_depth=[1,1])
        # lca_model.compile(optimizer='adam',
        #                   loss='categorical_crossentropy',
        #                   metrics=['accuracy'])
        lca_model.optimizer_allocation(lr=[.0001,.0001,.0001])

        lca_model.setup_lca_save(path='', basename='1D', occurence=10)
        lca_model.Fit(x=XT, y=YT, validation_data=(xt, yt), epochs=20,batch_size = 16)

    def test_epoch_return(self):
        XT, xt, YT, yt = load_data_covtype()
        # print(XT.shape,xt.shape,YT.shape,yt.shape)
        input = keras.Input(shape=(XT.shape[1]))
        X = keras.layers.Dense(5,activation='relu')(input)
        X = keras.layers.Dense(5,activation='relu')(X)
        # X = keras.layers.Dense(5)(X)
        X = keras.layers.Dense(5,activation='relu')(X)
        X = keras.layers.Dense(10,activation='relu')(X)
        out = keras.layers.Dense(YT.shape[1], activation='softmax')(X)
        lca_model = MultiSGDLCA(inputs=[input], outputs=out, lca_type='Mean',optimizer_allocation='edges',layer_depth=[1,1])
        # lca_model.compile(optimizer='adam',
        #                   loss='categorical_crossentropy',
        #                   metrics=['accuracy'])
        lca_model.optimizer_allocation(lr=[.0001,.0001,.0001])

        lca_model.setup_lca_save(path='', basename='1D', occurence=10)
        lca_model.Fit(x=XT, y=YT, validation_data=(xt, yt), epochs=2,batch_size = 16)
        assert type(lca_model.get_epoch_return_1d()) == list


if __name__ == '__main__':
    unittest.main()