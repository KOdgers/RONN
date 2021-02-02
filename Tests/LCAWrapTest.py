from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import unittest
import sys
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from Submodules.LettuceLeafs import LCAWrap



import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import fetch_covtype

# Helper libraries
import numpy as np
import pandas as pd


def load_data_covtype():
    print("Loading forest cover dataset...")

    cover = fetch_covtype()
    df = pd.DataFrame(cover['data'], columns=cover['feature_names'])
    target = cover['target']
    target=target-1
    target = to_categorical(target,7)

    return train_test_split(np.array(df)[:10000], np.array(target)[:10000],test_size=.3)

def load_fashion():
    print('TensorFlow version: {}'.format(tf.__version__))

    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_labels = to_categorical(train_labels,10)
    test_labels = to_categorical(test_labels,10)
    # scale the values to 0.0 to 1.0
    scale = np.max([np.max(train_images),np.max(test_images)])
    train_images = train_images / scale
    test_images = test_images / scale

    # reshape for feeding into the model
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)



    return train_images,test_images,train_labels,test_labels



class MyTestCase(unittest.TestCase):
    def test_model_2d_mean(self):
        XT,xt,YT,yt = load_fashion()
        XT = XT[:10000]
        YT = YT[:10000]
        # print(XT.shape,YT.shape)
        input = keras.Input(shape=(XT.shape[1],XT.shape[2],1))
        X = keras.layers.Conv2D(5,5,activation='relu')(input)
        X = keras.layers.BatchNormalization()(X)
        X = keras.layers.Dropout(.25)(X)
        X = keras.layers.Conv2D(5,5,activation='relu')(X)
        X = keras.layers.GlobalMaxPool2D()(X)
        X = keras.layers.Dense(100)(X)
        out = keras.layers.Dense(YT.shape[1],activation='softmax')(X)
        lca_model = LCAWrap(inputs=input,outputs=out,lca_type='Mean')
        lca_model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
        lca_model.setup_lca_save(path='', basename='2D', occurence=10)
        lca_model.Fit(x=XT,y=YT,validation_data=(xt,yt),epochs=2)
        # print(lca_model.LCA_vals)

    def test_model_1d_raw(self):
        XT,xt,YT,yt = load_data_covtype()
        # print(XT.shape,xt.shape,YT.shape,yt.shape)
        input = keras.Input(shape=(XT.shape[1]))
        X = keras.layers.Dense(5)(input)
        X = keras.layers.Dense(5)(X)
        X = keras.layers.Dense(10)(X)
        out = keras.layers.Dense(YT.shape[1], activation='softmax')(X)
        lca_model = LCAWrap(inputs=[input], outputs=out, lca_type='Raw')
        lca_model.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        lca_model.setup_lca_save(path='', basename='1D', occurence=10)
        lca_model.Fit(x=XT, y=YT, validation_data=(xt, yt), epochs=20)

    def test_model_1d_node(self):
        XT,xt,YT,yt = load_data_covtype()
        # print(XT.shape,xt.shape,YT.shape,yt.shape)
        input = keras.Input(shape=(XT.shape[1]))
        X = keras.layers.Dense(5)(input)
        X = keras.layers.Dense(5)(X)
        X = keras.layers.Dense(10)(X)
        out = keras.layers.Dense(YT.shape[1], activation='softmax')(X)
        lca_model = LCAWrap(inputs=[input], outputs=out, lca_type='Node')
        lca_model.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        lca_model.setup_lca_save(path='', basename='1D', occurence=10)
        lca_model.Fit(x=XT, y=YT, validation_data=(xt, yt), epochs=20)


    def test_memory_check(self):

        XT,xt,YT,yt = load_data_covtype()
        # print(XT.shape,xt.shape,YT.shape,yt.shape)
        input = keras.Input(shape=(XT.shape[1]))
        X = keras.layers.Dense(5,name="Dense1")(input)
        X = keras.layers.Dense(5,name="Dense2")(X)
        X = keras.layers.Dense(10,name="Dense3")(X)
        out = keras.layers.Dense(YT.shape[1], activation='softmax')(X)
        lca_model = LCAWrap(inputs=[input], outputs=out,layer_names=['Dense1','Dense2','Dense3'],lca_type='Mean')
        lca_model.setup_lca_save(path='', basename='1D', occurence=10)

        lca_model.check_memory(epochs=1)
        
    def test_memory_fail(self):
        XT,xt,YT,yt = load_data_covtype()
        # print(XT.shape,xt.shape,YT.shape,yt.shape)
        input = keras.Input(shape=(XT.shape[1]))
        X = keras.layers.Dense(100,name="Dense1")(input)
        X = keras.layers.Dense(100,name="Dense2")(X)
        X = keras.layers.Dense(100,name="Dense3")(X)
        out = keras.layers.Dense(YT.shape[1], activation='softmax')(X)
        lca_model = LCAWrap(inputs=[input], outputs=out,lca_type='Raw',layer_names=['Dense1','Dense2','Dense3'])
        lca_model.setup_lca_save(path='', basename='1D', occurence=10)

        self.assertRaises(Exception,lca_model.check_memory(),10000)


    def test_splitting(self):

        XT,xt,YT,yt = load_data_covtype()
        # print(XT.shape,xt.shape,YT.shape,yt.shape)
        input = keras.Input(shape=(XT.shape[1]))
        X = keras.layers.Dense(5)(input)
        X = keras.layers.Dense(5)(X)
        X = keras.layers.Dense(10)(X)
        out = keras.layers.Dense(YT.shape[1], activation='softmax')(X)
        lca_model = LCAWrap(inputs=[input], outputs=out)
        lca_model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
        lca_model.setup_lca_save(path='', basename='1D', occurence=10)
        lca_model.Fit(x=XT,y=YT,validation_split=.2,epochs=2)
        # print(lca_model.last_LCA)
        # print(lca_model.LCA_vals)
    #
    # def test_model_lca_storage(self):
    #
    #
    # def test_lca_saving(self):


if __name__ == '__main__':
    unittest.main()
