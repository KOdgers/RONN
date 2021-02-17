import sys
import warnings
sys.path.append('.')

import tensorflow as tf
from tensorflow import keras
import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import BoundedArraySpec
from tf_agents.trajectories import time_step
import tensorflow_datasets as tfds
from sklearn.datasets import fetch_covtype, fetch_kddcup99, fetch_openml


from Submodules.SaladWrap import *
from Submodules.DataLoaders import *
from Submodules.NetworkExamples import *

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

    def add_data(self, Data,name = 'NA'):
        if self.XT:
            print('Data already loaded')
        else:
            XT, xt, YT, yt = Data['XT'], Data['xt'], Data['YT'], Data['yt']
            self.XT = XT
            self.YT = YT
            self.xt = xt
            self.yt = yt
            self.data_name=name

    def add_model(self, model):
        if self.model:
            return ('Model already loaded')
        else:
            if self.data_name not in model.valid_data:
                warnings.warn('Data and model are not recommended')
            self.model = model

    def initialize_bar(self, data=None, model=None, max_epochs=100):
        self.add_data(data)
        self.add_model(model)
        self.max_epochs = max_epochs

    def reset_bar(self):
        inp, out = self.model.build(self.XT.shape[1], self.YT.shape[1])
        self.SaladWrap = MultiSGDLCA(inputs=[inp], outputs=out, lca_type='Mean', optimizer_allocation=self.params['lr'])
        self.SaladWrap.optimizer_allocation()
        self.SaladWrap.Fit(x=self.XT, y=self.YT, validation_data=(self.xt, self.yt), epochs=1, batch_size=64)
        self.last_loss = self.SaladWrap.loss




    def advance(self, opts):
        self.SaladWrap.optimizer_allocation(opts)
        self.SaladWrap.Fit(x=self.XT, y=self.YT, validation_data=(self.xt, self.yt), batch_size=32, epochs=1)
        reward = self.reward_function()

        return np.asarray(reward, dtype=np.float32)

    # def reward_function(self):
    #     return ((self.SaladWrap.loss-self.last_loss) * len(self.SaladWrap.layers))# -
                # np.sum([item / np.abs(item) for item in self.SaladWrap.get_lca_1d() if item != 0]))

    def reward_function(self):
        lca = self.SaladWrap.get_lca_1d()
        return(-1*np.sum([item / np.abs(item) if item != 0 else 0 for item in lca])/len(lca))
