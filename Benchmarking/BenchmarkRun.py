

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorboard.plugins.hparams import api as hp
import sys
import datetime


import base64
import imageio
import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import kerastuner as kt

tf.compat.v1.enable_v2_behavior()


sys.path.append('.')
from Submodules.NetworkExamples import *
from Submodules.DataLoaders import *

class Benchmarking:

    def __init__(self,model,data, max_epochs = 50, max_tests = 5):
        self.data_loader = data
        self.model_builder = model
        self.max_epochs = max_epochs
        self.max_tests = max_tests
        self.title = data.__name__+'___'+model.name

        HP_Optimizer = hp.HParam('optimizer',hp.Discrete(['adam','sgd']))
        HP_LR = hp.HParam('learning_rate', hp.RealInterval(.0000001,1.0))
        Metric_Accuracy = 'categorical_accuracy'
        self.hparams = hp.hparams_config(
            hparams=[HP_Optimizer, HP_LR],
            metrics = [hp.Metric(Metric_Accuracy, display_name = 'categorical_accuracy')]
        )

    def Optimize(self):
        Data = self.data_loader('train')
        self.XT = Data['XT']
        self.YT = Data['YT']
        self.xt = Data['xt']
        self.yt = Data['yt']
        # inp, outp = self.model_builder.build(self.XT.shape[1], self.YT.shape[1])
        # self.Model = keras.models.Model(inputs = inp, outputs = outp)

        def model_builder(hp):
            inp, outp = self.model_builder.build(Data['XT'].shape[1], Data['YT'].shape[1])
            mymodel = keras.models.Model(inputs=inp, outputs=outp)
            hp_lr = hp.Float('LR', min_value=-7, max_value=0)
            hp_optimizer = hp.Choice('optimizer', values=['adam', 'adagrad', 'SGD'])
            if hp_optimizer == 'adam':
                opt = keras.optimizers.Adam(learning_rate=10 ** hp_lr)
            elif hp_optimizer == 'adagrad':
                opt = keras.optimizers.Adagrad(learning_rate=10 ** hp_lr)
            else:
                opt = keras.optimizers.SGD(learning_rate=10 ** hp_lr)
            mymodel.compile(optimizer=opt,
                            loss='categorical_crossentropy', metrics='categorical_accuracy')
            return mymodel



        tuner = kt.Hyperband(model_builder, objective='val_categorical_accuracy',
                             max_epochs=20, factor=3,
                             directory='hplog', project_name=self.title,
                             overwrite = True)
        se_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

        tuner.search(self.XT, self.YT, epochs=50, validation_split=0.2, callbacks=[se_callback],verbose=0)

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]


        return best_hps.get('optimizer'), best_hps.get('LR')

    def Evaluate(self):
        def model_builder(hp):
            inp, outp = self.model_builder.build(self.XT.shape[1], self.YT.shape[1])
            mymodel = keras.models.Model(inputs=inp, outputs=outp)
            hp_lr = hp.Float('LR', min_value=-7, max_value=0)
            hp_optimizer = hp.Choice('optimizer', values=['adam', 'adagrad', 'SGD'])
            if hp_optimizer == 'adam':
                opt = keras.optimizers.Adam(learning_rate=10 ** hp_lr)
            elif hp_optimizer == 'adagrad':
                opt = keras.optimizers.Adagrad(learning_rate=10 ** hp_lr)
            else:
                opt = keras.optimizers.SGD(learning_rate=10 ** hp_lr)
            mymodel.compile(optimizer=opt,
                            loss='categorical_crossentropy', metrics='categorical_accuracy')
            return mymodel

        tuner = kt.Hyperband(model_builder, objective='val_categorical_accuracy',
                             max_epochs=20, factor=4,
                             directory='hplog', project_name=self.title,
                             overwrite=True)

        hypermodel = tuner.hypermodel.build(self.best_hps)
        hypermodel.fit(self.XT, self.YT, epochs=30)
        eval_result = hypermodel.evaluate(self.xt, self.yt)
        return eval_result[0], eval_result[1]
