import sys
sys.path.append('.')
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import categorical_accuracy
from sklearn.utils import shuffle

from Submodules.LettuceLeafs import *


class MultiSGDLCA(LCAWrap):

    def __init__(self,**kwargs):

        if 'optimizer_allocation' not in kwargs.keys():
            self.optimizer_allocation = 'all'
        else:
            self.optimizer_allocation = kwargs['optimizer_allocation']
            del kwargs['optimizer_allocation']
            if 'layer_depth' in kwargs.keys():
                self.layer_depths = kwargs['layer_depth']
                del kwargs['layer_depth']

        super().__init__(**kwargs)

        self.group_optimizers()
        self.epochs_run = 0

    def group_optimizers(self):
        if type(self.optimizer_allocation) == list:
            self.optimizer_allocation = self.per_layer_optimizer
        elif self.optimizer_allocation =='all':
            self.optimizer_allocation = self.single_opt
        elif self.optimizer_allocation  =='edges':
            self.optimizer_allocation = self.edge_opts

    def single_opt(self,lr=.0001):
        opt = Adam(learning_rate=lr)
        self.opt_layers = {'opt1':self.layer_names}
        self.opt = {'opt1':opt}

    def edge_opts(self,lr=[.0001,.0001,.0001]):
        self.opt = {'opt1':Adam(learning_rate=lr[0]),'opt2':Adam(learning_rate=lr[1]),
                    'opt2':Adam(learning_rate=lr[2])}

        self.opt_layers={}
        for i in range(0,len(self.layer_names)):
            if i <=self.layer_depths[0]:
                temp_opt = 'opt1'
            elif i >(len(self.layer_names)-self.layer_depths[1]):
                temp_opt = 'opt3'
            else:
                temp_opt='opt2'
            try:
                self.opt_layers[temp_opt].append(self.layer_names[i])
            except:
                self.opt_layers[temp_opt]=[self.layer_names[i]]

    def per_layer_optimizer(self,lr=[]):
        assert len(lr) == len(self.layer_names), "Bad Learning Rate List!"

        self.opt = {}
        self.opt_layers={}
        for i in range(0,len(lr)):
            self.opt['opt'+str(i+1)] = Adam(learning_rate=lr[i])
            self.opt_layers['opt'+str(i+1)] = [self.layer_names[i]]

    def single_epoch_run(self,**kwargs):
        XTrain = kwargs['x']
        YTrain = kwargs['y']
        # build our model and initialize our optimizer

        numUpdates = int(XTrain.shape[0] / self.batch_size)
        XTrain, YTrain = shuffle(XTrain, YTrain)
        # loop over the number of epochs
        def step(X, y):

            with tf.GradientTape() as tape:
                pred = self(X)
                loss = categorical_crossentropy(y, pred)

            grads = tape.gradient(loss, self.trainable_variables)
            # grads1 = grads[::2]
            # grads2 = grads[1::2]
            # self.opt.apply_gradients(zip(grads, self.Model.trainable_variables))
            # self.old
            for j in self.opt.keys():
                # print(len([grads[i] for i in range(0,len(grads))
                #                                  if self.trainable_numbers[i].split('/')[0] in self.opt_layers[j]]))
                self.opt[j].apply_gradients(zip([grads[i] for i in range(0,len(grads))
                                                 if self.trainable_numbers[i].split('/')[0] in self.opt_layers[j]],
                                                [self.trainable_variables[i] for i in range(0,len(grads))
                                                 if self.trainable_numbers[i].split('/')[0] in self.opt_layers[j]]))

            # tf.group(self.opt.values)


        for i in range(0, numUpdates):

            start_index = i * self.batch_size
            end_index = start_index + self.batch_size

            step(XTrain[start_index:end_index], YTrain[start_index:end_index])

        pred = self(self.x_test)
        acc = categorical_accuracy(self.y_test, pred)

        print(np.mean(acc))
        self.loss = np.mean(acc)
        #


    def Fit(self,**kwargs):
        try:
            epochs = kwargs['epochs']
            del kwargs['epochs']
        except:
            epochs =1

        try :
            self.batch_size = kwargs['batch_size']
        except:
            self.batch_size = 16
        for i in range(0,epochs):
            self.epochs_run +=1
            super().Fit(new_fitting=self.single_epoch_run,epochs =1,**kwargs)
            # print(self.last_LCA)
        # return self.last_LCA, self.loss


    def get_epoch_return_1d(self):
        epoch_out = []
        for i in (self.last_LCA.keys()):
            # epoch_out.append(self.last_LCA[i])
            templ = float(self.last_LCA[i])
            if templ:
                templ = templ / np.abs(templ) * min([1000, np.abs(templ)])

            epoch_out.append((templ if templ is not np.nan else 0))
        for i in list(self.opt.keys()):
            epoch_out.append(float(self.opt[i].learning_rate.value()))
        return np.asarray(epoch_out, dtype=np.float32)

    def get_lca_1d(self):
        epoch_out = []
        for i in (self.last_LCA.keys()):
            templ = float(self.last_LCA[i])
            if templ:
                templ = templ/np.abs(templ)*min([1000,np.abs(templ)])

            epoch_out.append((templ if templ is not np.nan else 0))
        return np.asarray(epoch_out)




