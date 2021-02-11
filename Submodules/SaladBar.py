import sys
sys.path.append('.')

import tensorflow as tf
from tensorflow import keras
import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import BoundedArraySpec
from tf_agents.trajectories import time_step
import tensorflow_datasets as tfds
from sklearn.datasets import fetch_covtype, fetch_kddcup99

from Submodules.SaladWrap import *

tf.compat.v1.enable_v2_behavior()


class SaladBar:

    def __init__(self):
        self.XT = None
        self.YT = None
        self.xt = None
        self.yt = None
        self.model = None
        self.max_epochs = None
        self.params = {}
        self.params['lr'] = 'individual'

    def add_data(self, Data):
        if self.XT:
            print('Data already loaded')
        else:
            XT, xt, YT, yt = Data['XT'], Data['xt'], Data['YT'], Data['yt']
            self.XT = XT
            self.YT = YT
            self.xt = xt
            self.yt = yt

    def add_model(self, model):
        if self.model:
            return ('Model already loaded')
        else:
            self.model = model

    def initialize_bar(self, data=None, model=None, max_epochs=100):
        self.add_data(data)
        self.add_model(model)
        self.max_epochs = max_epochs

    def reset_bar(self):
        inp, out = self.model(self.XT.shape[1], self.YT.shape[1])
        self.SaladWrap = MultiSGDLCA(inputs=[inp], outputs=out, lca_type='Mean', optimizer_allocation=self.params['lr'])
        self.SaladWrap.optimizer_allocation()
        self.SaladWrap.Fit(x=self.XT, y=self.YT, validation_data=(self.xt, self.yt), epochs=1, batch_size=64)
        self.last_loss = self.SaladWrap.loss

    def load_data_covtype(self, type='train'):
        print("Loading forest cover dataset...")

        cover = fetch_covtype()
        df = pd.DataFrame(cover['data'], columns=cover['feature_names'])
        target = cover['target']
        target = target - 1
        target = to_categorical(target, 7)
        if type == 'train':
            XT, xt, YT, yt = train_test_split(np.asarray(df).astype('float32')[:15000], np.array(target)[:15000], test_size=.3)
        else:
            XT, xt, YT, yt = train_test_split(np.asarray(df).astype('float32')[20000:40000], np.array(target)[20000:40000], test_size=.3)
        return {'XT': XT, 'YT': YT, 'xt': xt, 'yt': yt}

    def load_data_kddcup(self,type='train'):
        from sklearn.preprocessing import LabelEncoder
        print("Loading kddcup dataset...")

        cover = fetch_kddcup99()
        df = pd.DataFrame(cover['data'], columns=cover['feature_names'])
        df['protocol_type'] = LabelEncoder().fit(list(set(df['protocol_type']))).transform(df['protocol_type'])
        df['service'] = LabelEncoder().fit(list(set(df['service']))).transform(df['service'])
        df['flag'] = LabelEncoder().fit(list(set(df['flag']))).transform(df['flag'])

        target = cover['target']
        target = [item.decode("utf-8") for item in target]
        le = LabelEncoder()
        le.fit(list(set(target)))
        target = le.transform(target)
        target = to_categorical(target,23)
        # target = target - 1
        # target = to_categorical(target, 23,dtype='str')
        if type == 'train':
            XT, xt, YT, yt = train_test_split(np.array(df)[:15000], np.array(target)[:15000], test_size=.3)
        else:
            XT, xt, YT, yt = train_test_split(np.array(df)[20000:40000], np.array(target)[20000:40000], test_size=.3)
        return {'XT': XT, 'YT': YT, 'xt': xt, 'yt': yt}


    def load_test_model(self, inshape, outshape):
        input = keras.Input(shape=(inshape))
        X = keras.layers.Dense(5, activation='relu')(input)
        X = keras.layers.Dense(5, activation='relu')(X)
        # X = keras.layers.Dense(5, activation='relu')(X)
        # X = keras.layers.Dense(10, activation='relu')(X)
        out = keras.layers.Dense(outshape, activation='softmax')(X)

        return input, out

    def advance(self, opts):
        self.SaladWrap.optimizer_allocation(opts)
        self.SaladWrap.Fit(x=self.XT, y=self.YT, validation_data=(self.xt, self.yt), batch_size=32, epochs=1)
        reward = self.reward_function()

        return np.asarray(reward, dtype=np.float32)

    # def reward_function(self):
    #     return ((self.SaladWrap.loss-self.last_loss) * len(self.SaladWrap.layers))# -
                # np.sum([item / np.abs(item) for item in self.SaladWrap.get_lca_1d() if item != 0]))

    def reward_function(self):
        return(-1*np.sum([item / np.abs(item) if item != 0 else 0 for item in self.SaladWrap.get_lca_1d()]))
