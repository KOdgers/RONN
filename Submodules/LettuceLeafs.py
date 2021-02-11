import sys

from psutil import virtual_memory

import numpy as np
import tensorflow as tf
import pandas as pd
import tensorflow.keras.optimizers as opts
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import categorical_accuracy
from sklearn.utils import shuffle
from tensorflow.python.util import tf_decorator

from sklearn.model_selection import train_test_split


class LCAWrap(Model):

    def __init__(self, *args, **kwargs):

        try:
            self.layer_names = kwargs['layer_names']

            del kwargs['layer_names']
        except:
            self.layer_names = None

        try:
            self.lca_type = kwargs['lca_type']
            del kwargs['lca_type']
        except:
            self.lca_type = 'Mean'

        super(LCAWrap, self).__init__(**kwargs)
        self.LCA_vals = {}
        self.last_LCA = None
        self.memory_threshold = .9
        self.lca_save = False
        self.df = None
        self.max_occurence = None
        self.variable_names = None

        self.Weights = False
        self.OldWeights = False

        if not self.layer_names:
            self.get_layer_names()
            self.get_variable_names()
        else:
            self.get_variable_names()

        self.variable_index()

    def setup_lca_save(self, path, basename, occurence):
        self.path = path
        self.basename = basename
        self.max_occurence = occurence
        self.lca_save = True

    def Fit(self, new_fitting=None, **kwargs):

        self.check_memory()
        epochs = kwargs['epochs']
        del kwargs['epochs']

        self.OldWeights = self.get_weights()
        if 'validation_data' not in kwargs.keys():
            X_train, x_test, Y_train, y_test = train_test_split(kwargs['x'], kwargs['y'],
                                                                test_size=kwargs['validation_split'])
            kwargs['x'] = X_train
            kwargs['y'] = Y_train
            kwargs['validation_data'] = (x_test, y_test)

            del kwargs['validation_split']
        else:
            x_test = kwargs['validation_data'][0]
            y_test = kwargs['validation_data'][1]
        self.x_test = np.asarray(x_test).astype('float32')
        self.y_test = y_test
        # print('Y_test shape:',y_test.shape)
        self.epochs = epochs
        for j in range(0, epochs):
            self.current_epoch = j
            assert self.single_epoch_size < virtual_memory()[1], "LCA will fill into swap this epoch"

            if not new_fitting:
                self.fit(**kwargs, epochs=1)
            else:
                new_fitting(**kwargs, epoch=1)
            self.get_grads()
            self.Weights = self.get_weights()

            self.last_LCA = self.get_LCA()
            self.OldWeights = self.Weights

        # print(self.LCA_vals)

    # def lca_out(self,path='',name = 'Temporary.h5'):
    #     temp_dict = self.LCA_vals
    #     temp_df = pd.DataFrame.from_dict(temp_dict)
    #     temp_df.to_hdf(path+name,key=str(self.current_epoch))
    #     self.LCA_vals = {}

    def lca_stream(self, path='', basename='', save_occurence=None, lca=None):
        # temp_df = pd.DataFrame(lca,index=[self.current_epoch])
        temp_df = pd.DataFrame(columns=lca.keys(), dtype=object)
        for item in lca.keys():
            temp_df.loc[self.current_epoch, item] = lca[item]
        if self.df is not None:
            self.df = pd.concat([self.df, temp_df])
        else:
            self.df = temp_df

        if (
                self.current_epoch % save_occurence == 0 and self.current_epoch != 0) or self.current_epoch == self.epochs - 1:
            self.df.to_hdf(path + basename + str(int(self.current_epoch - save_occurence)) + '.h5',
                           key=(str(self.current_epoch - save_occurence)))
            self.df = None

    def get_grads(self):
        with tf.GradientTape() as tape:
            pred = self(self.x_test)
            loss = categorical_crossentropy(self.y_test, pred)
        return tape.gradient(loss, self.trainable_variables)

    def variable_index(self):
        listOfVariableTensors = self.trainable_weights
        self.trainable_numbers = {}

        for l in range(0, len(listOfVariableTensors)):
            if listOfVariableTensors[l].name in self.variable_names:
                self.trainable_numbers[l] = listOfVariableTensors[l].name

    def get_weights(self):
        listOfVariableTensors = self.trainable_weights
        Weights = {}

        for l in range(0, len(listOfVariableTensors)):
            if listOfVariableTensors[l].name in self.variable_names:
                Weights[listOfVariableTensors[l].name] = listOfVariableTensors[l].value()

        return Weights

    def calculate_mean_LCA(self):
        if not self.OldWeights:
            return 'Model hasnt been run or oldweights have been lost'
        grads = self.get_grads()
        LCA = {}
        for j, name in enumerate(self.variable_names):
            lca = grads[list(self.trainable_numbers.keys())[j]] * (self.Weights[name] - self.OldWeights[name])

            LCA[name] = np.array(np.mean(lca))
        if self.lca_save:
            self.lca_stream(path=self.path, basename=self.basename, save_occurence=self.max_occurence, lca=LCA)
        return LCA

    def calculate_LCA(self):
        if not self.OldWeights:
            return 'Model hasnt been run or oldweights have been lost'
        grads = self.get_grads()
        LCA = {}
        for j, name in enumerate(self.variable_names):
            lca = grads[list(self.trainable_numbers.keys())[j]] * (self.Weights[name] - self.OldWeights[name])

            LCA[name] = np.array(lca)
        if self.lca_save:
            self.lca_stream(path=self.path, basename=self.basename, save_occurence=self.max_occurence, lca=LCA)
        return LCA

    def calculate_per_node_LCA(self):
        if not self.OldWeights:
            return 'Model hasnt been run or oldweights have been lost'
        grads = self.get_grads()
        LCA = {}
        for j, name in enumerate(self.variable_names):
            lca = grads[list(self.trainable_numbers.keys())[j]] * (self.Weights[name] - self.OldWeights[name])

            LCA[name] = np.mean(lca, axis=0)
        if self.lca_save:
            self.lca_stream(path=self.path, basename=self.basename, save_occurence=self.max_occurence, lca=LCA)
        return LCA

    def get_LCA(self):
        if self.lca_type == 'Mean':
            return self.calculate_mean_LCA()
        elif self.lca_type == 'Raw':
            return self.calculate_LCA()
        elif self.lca_type == 'Node':
            return self.calculate_per_node_LCA()

    def get_layer_names(self):
        self.layer_names = []
        for layer in self.layers:
            if layer.trainable:
                self.layer_names.append(layer.name)
        self.variable_names = [item.name for item in self.trainable_weights]

    def get_variable_names(self):
        variable_names = [item.name for item in self.trainable_weights]
        self.variable_names = []
        for item in variable_names:
            if item.split('/')[0] in self.layer_names:
                self.variable_names.append(item)

    def check_memory(self, epochs=1):

        tempArray = self.get_weights()
        if self.lca_type == 'Mean':
            templist = []
            for j in tempArray:
                templist.append(np.mean(tempArray[j]))
            size = sys.getsizeof(templist)
        elif self.lca_type == 'Raw':
            size = sys.getsizeof(tempArray)
        elif self.lca_type == 'Node':
            templist = []
            for j in tempArray:
                templist.append(np.mean(tempArray[j], axis=0))
            size = sys.getsizeof(templist)
        self.single_epoch_size = size * 3
        total_size = size * (epochs + 3)
        available_space = virtual_memory()[1]
        try:
            self.max_occurence = min(available_space // (size * 3), self.max_occurence)
        except:
            self.max_occurence = available_space // (size * 3)
        if self.max_occurence == 0:
            raise Exception(" Single Epoch LCA will fill memory")
        if total_size > self.memory_threshold * available_space:
            raise Exception("LCA will fill memory before completing epochs")




#
# import numpy as np
# import tensorflow as tf
# import pandas as pd
# import sys
# from psutil import virtual_memory
# import tensorflow.keras.optimizers as opts
# from tensorflow.keras.losses import categorical_crossentropy
# from tensorflow.keras.models import Model
# from tensorflow.keras.utils import to_categorical
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.metrics import categorical_accuracy
# from sklearn.utils import shuffle
#
#
#
# from tensorflow.python.util import tf_decorator
#
# from sklearn.model_selection import train_test_split
#
# class LCAWrap(Model):
#
#     def __init__(self,**kwargs):
#
#         try:
#             self.layer_names = kwargs['layer_names']
#
#             del kwargs['layer_names']
#         except:
#             self.layer_names = None
#
#
#
#         try:
#             self.lca_type = kwargs['lca_type']
#             del kwargs['lca_type']
#         except:
#             self.lca_type = 'Mean'
#
#
#
#         super(LCAWrap, self).__init__(**kwargs)
#         self.LCA_vals = {}
#         self.last_LCA = None
#         self.memory_threshold = .9
#         self.lca_save = False
#         self.df = None
#         self.max_occurence=None
#         self.variable_names = None
#
#         self.Weights=False
#         self.OldWeights = False
#
#         if not self.layer_names:
#             self.get_layer_names()
#             self.get_variable_names()
#         else:
#             self.get_variable_names()
#
#     def setup_lca_save(self,path,basename,occurence):
#         self.path = path
#         self.basename = basename
#         self.max_occurence=occurence
#         self.lca_save=True
#
#
#     def Fit(self,new_fitting = None,**kwargs):
#
#
#         self.check_memory()
#         epochs = kwargs['epochs']
#         del kwargs['epochs']
#
#         self.OldWeights = self.get_weights()
#         if 'validation_data' not in kwargs.keys():
#             X_train,x_test,Y_train,y_test = train_test_split(kwargs['x'],kwargs['y'],test_size=kwargs['validation_split'])
#             kwargs['x']=X_train
#             kwargs['y']=Y_train
#             kwargs['validation_data'] = (x_test,y_test)
#
#             del kwargs['validation_split']
#         else:
#             x_test= kwargs['validation_data'][0]
#             y_test = kwargs['validation_data'][1]
#         self.x_test = x_test
#         self.y_test = y_test
#         # print('Y_test shape:',y_test.shape)
#         self.epochs = epochs
#         for j in range(0,epochs):
#             self.current_epoch = j
#             assert self.single_epoch_size<virtual_memory()[1], "LCA will fill into swap this epoch"
#
#             if not new_fitting:
#                 self.fit(**kwargs,epochs=1)
#             else:
#                 new_fitting(**kwargs)
#             self.get_grads()
#             self.Weights = self.get_weights()
#
#             self.last_LCA = self.get_LCA()
#             self.OldWeights = self.Weights
#
#         # print(self.LCA_vals)
#
#
#
#     # def lca_out(self,path='',name = 'Temporary.h5'):
#     #     temp_dict = self.LCA_vals
#     #     temp_df = pd.DataFrame.from_dict(temp_dict)
#     #     temp_df.to_hdf(path+name,key=str(self.current_epoch))
#     #     self.LCA_vals = {}
#
#     def lca_stream(self,path='',basename='',save_occurence=None,lca=None):
#         # temp_df = pd.DataFrame(lca,index=[self.current_epoch])
#         temp_df = pd.DataFrame(columns=lca.keys(),dtype = object)
#         for item in lca.keys():
#             temp_df.loc[self.current_epoch,item] = lca[item]
#         if self.df is not None:
#             self.df = pd.concat([self.df, temp_df])
#         else:
#             self.df = temp_df
#
#         if (self.current_epoch%save_occurence ==0 and self.current_epoch!=0) or self.current_epoch==self.epochs-1:
#             self.df.to_hdf(path+basename+str(int(self.current_epoch-save_occurence))+'.h5',key=(str(self.current_epoch-save_occurence)))
#             self.df = None
#
#
#     def get_grads(self):
#         with tf.GradientTape() as tape:
#             pred = self(self.x_test)
#             loss = categorical_crossentropy(self.y_test,pred)
#         self.last_loss =loss
#         return tape.gradient(loss,self.trainable_variables)
#
#
#     def get_weights(self):
#         listOfVariableTensors = self.trainable_weights
#         Weights = {}
#         self.trainable_numbers = []
#
#         for l in range(0, len(listOfVariableTensors)):
#             if listOfVariableTensors[l].name in self.variable_names:
#                 self.trainable_numbers.append(l)
#                 Weights[listOfVariableTensors[l].name]=listOfVariableTensors[l].value()
#
#         return Weights
#
#     def calculate_mean_LCA(self):
#         if not self.OldWeights:
#             return 'Model hasnt been run or oldweights have been lost'
#         grads = self.get_grads()
#         LCA = {}
#         for j,name in enumerate(self.variable_names):
#             lca = grads[self.trainable_numbers[j]]*(self.Weights[name]-self.OldWeights[name])
#
#             LCA[name]=np.array(np.mean(lca))
#         if self.lca_save:
#             self.lca_stream(path = self.path,basename = self.basename,save_occurence = self.max_occurence,lca=LCA)
#         return LCA
#
#
#     def calculate_LCA(self):
#         if not self.OldWeights:
#             return 'Model hasnt been run or oldweights have been lost'
#         grads = self.get_grads()
#         LCA = {}
#         for j,name in enumerate(self.variable_names):
#             lca = grads[self.trainable_numbers[j]]*(self.Weights[name]-self.OldWeights[name])
#
#             LCA[name]=np.array(lca)
#         if self.lca_save:
#             self.lca_stream(path=self.path, basename=self.basename, save_occurence=self.max_occurence, lca=LCA)
#         return LCA
#
#     def calculate_per_node_LCA(self):
#         if not self.OldWeights:
#             return 'Model hasnt been run or oldweights have been lost'
#         grads = self.get_grads()
#         LCA = {}
#         for j,name in enumerate(self.variable_names):
#             lca = grads[self.trainable_numbers[j]]*(self.Weights[name]-self.OldWeights[name])
#
#             LCA[name]=np.mean(lca,axis=0)
#         if self.lca_save:
#             self.lca_stream(path=self.path, basename=self.basename, save_occurence=self.max_occurence, lca=LCA)
#         return LCA
#
#     def get_LCA(self):
#         if self.lca_type=='Mean':
#             return self.calculate_mean_LCA()
#         elif self.lca_type=='Raw':
#             return self.calculate_LCA()
#         elif self.lca_type=='Node':
#             return self.calculate_per_node_LCA()
#
#     def get_layer_names(self):
#         self.layer_names=[]
#         for layer in self.layers:
#             if layer.trainable:
#                 self.layer_names.append(layer.name)
#         self.variable_names = [item.name for item in self.trainable_weights]
#
#     def get_variable_names(self):
#         variable_names = [item.name for item in self.trainable_weights]
#         self.variable_names = []
#         for item in variable_names:
#             if item.split('/')[0] in self.layer_names:
#                 self.variable_names.append(item)
#
#     def check_memory(self,epochs=1):
#
#         tempArray = self.get_weights()
#         if self.lca_type=='Mean':
#             templist = []
#             for j in tempArray:
#                 templist.append(np.mean(tempArray[j]))
#             size = sys.getsizeof(templist)
#         elif self.lca_type =='Raw':
#             size = sys.getsizeof(tempArray)
#         elif self.lca_type =='Node':
#             templist = []
#             for j in tempArray:
#                 templist.append(np.mean(tempArray[j],axis=0))
#             size = sys.getsizeof(templist)
#         self.single_epoch_size = size*3
#         total_size = size*(epochs+3)
#         available_space = virtual_memory()[1]
#         try:
#             self.max_occurence = min(available_space//(size*3),self.max_occurence)
#         except:
#             self.max_occurence = available_space//(size*3)
#         if self.max_occurence==0:
#             raise Exception(" Single Epoch LCA will fill memory")
#         if total_size > self.memory_threshold*available_space:
#             raise Exception("LCA will fill memory before completing epochs")
#
#
#
#
#
