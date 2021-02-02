import tensorflow as tf
from tensorflow import keras

from tf_agents.environments import py_environment

from tf_agents.specs import BoundedArraySpec

from tf_agents.trajectories import time_step
from sklearn.datasets import fetch_covtype

tf.compat.v1.enable_v2_behavior()


from Submodules.SaladWrap import *
import numpy as np


class SaladBar:

    def __init__(self):
        self.XT = None
        self.YT = None
        self.xt = None
        self.yt = None
        self.model = None
        self.max_epochs = None
        self.params={}
        self.params['lr']='all'

    def add_data(self,Data):
        if self.XT:
            print('Data already loaded')
        else:
            XT, xt, YT, yt = Data['XT'],Data['xt'],Data['YT'],Data['yt']
            self.XT = XT
            self.YT = YT
            self.xt = xt
            self.yt = yt

    def add_model(self, model):
        if self.model:
            return('Model already loaded')
        else:
            self.model = model

    def initialize_bar(self,data=None, model=None, max_epochs=100):
        self.add_data(data)
        self.add_model(model)
        self.max_epochs = max_epochs

    def reset_bar(self):
        inp,out = self.model(self.XT.shape[1],self.YT.shape[1])
        self.SaladWrap = MultiSGDLCA(inputs = [inp],outputs=out,lca_type='Mean',optimizer_allocation=self.params['lr'])
        self.SaladWrap.optimizer_allocation()
        self.SaladWrap.Fit(x=self.XT, y=self.YT, validation_data=(self.xt, self.yt), epochs=1,batch_size=64)
        self.last_loss = self.SaladWrap.loss

    def load_data_covtype(self,type='train'):
        print("Loading forest cover dataset...")

        cover = fetch_covtype()
        df = pd.DataFrame(cover['data'], columns=cover['feature_names'])
        target = cover['target']
        target = target - 1
        target = to_categorical(target, 7)
        if type == 'train':
            XT, xt, YT, yt = train_test_split(np.array(df)[:10000], np.array(target)[:10000], test_size=.3)
        else:
            XT, xt, YT, yt = train_test_split(np.array(df)[10000:20000], np.array(target)[10000:20000], test_size=.3)
        return {'XT': XT, 'YT': YT, 'xt': xt, 'yt': yt}

    def load_test_model(self,inshape, outshape):
        input = keras.Input(shape=(inshape))
        X = keras.layers.Dense(5, activation='relu')(input)
        X = keras.layers.Dense(5, activation='relu')(X)
        X = keras.layers.Dense(5, activation='relu')(X)
        X = keras.layers.Dense(10, activation='relu')(X)
        out = keras.layers.Dense(outshape, activation='softmax')(X)

        return input, out

    def advance(self,opts):
        self.SaladWrap.optimizer_allocation(opts)
        self.SaladWrap.Fit(x=self.XT,y=self.YT,validation_data = (self.xt,self.yt),batch_size=16,epochs = 1)
        reward = self.reward_function()

        return np.asarray(reward,dtype=np.float32)

    def reward_function(self):
        return ((self.last_loss-self.SaladWrap.loss)*len(self.SaladWrap.layers) +
                np.sum([item/np.abs(item) for item in self.SaladWrap.get_lca_1d() if item !=0]))





class RONNEnviron1D(py_environment.PyEnvironment):

    def __init__(self,train_eval='train'):
        # super().__init__()
        self.train_eval = train_eval

        self.SaladBar = SaladBar()
        self.SaladBar.add_data(self.SaladBar.load_data_covtype(self.train_eval))
        self.SaladBar.add_model(self.SaladBar.load_test_model)
        self.max_epochs = 100
        self.SaladBar.reset_bar()


        self._observation_spec = BoundedArraySpec(shape=(len(self.SaladBar.SaladWrap.get_epoch_return_1d()),),
                                                  dtype=np.float32,
                                                  minimum=[-1000]*len(self.SaladBar.SaladWrap.get_epoch_return_1d()),
                                                  maximum=[1000]*len(self.SaladBar.SaladWrap.get_epoch_return_1d()),
                                                  name='observation')
        #
        self._action_spec = BoundedArraySpec(shape=(),
                                             dtype=np.float32,
                                             name='action',
                                             minimum=0,
                                             maximum=1)
        # self._reward_spec = BoundedArraySpec( shape = (1,),
        #                                       dtype=np.float32,
        #                                       minimum=-10,
        #                                       maximum=10, name='reward')
        # self._discount_spec = BoundedArraySpec(shape=(1,), dtype=np.float32, minimum=0, maximum=1,
        #                                                   name='discount')
        #


    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    # def reward_spec(self):
    #     return self._reward_spec
    #
    # def discount_spec(self):
    #     return self._discount_spec

    def _step(self, action):
        # if self._episode_ended:
        #     return self.reset()
        """Apply action and return new time_step."""
        # if self._current_time_step is None:
        #     return self.reset()
        # self._current_time_step = self._step(action)
        reward = self.SaladBar.advance(action)
        # print(type(reward))
        print(self.SaladBar.SaladWrap.loss, self.SaladBar.last_loss)
        print(self.SaladBar.SaladWrap.epochs_run, self.max_epochs)

        if ((self.SaladBar.SaladWrap.loss > self.SaladBar.last_loss) or
                (self.SaladBar.SaladWrap.epochs_run == self.max_epochs)):
            self._episode_ended =True
            print('Reward:',reward)
            return time_step.termination(observation=self.SaladBar.SaladWrap.get_epoch_return_1d(),
                                         reward=reward
                                         )
        else:
            self.SaladBar.last_loss=self.SaladBar.SaladWrap.loss
            return time_step.transition(observation=self.SaladBar.SaladWrap.get_epoch_return_1d(),
                                        reward=reward,discount=1.0
                                        )
        # return self._current_time_step

    # def reset(self):
    #     return self._reset()

    def _reset(self):
        """Return initial_time_step."""
        self._episode_ended=False
        self.SaladBar.reset_bar()
        return time_step.restart(observation=self.SaladBar.SaladWrap.get_epoch_return_1d())



# env = RONNEnviron1D(train_eval='train')
# # env = CardGameEnv()
# # env = tf_py_environment.TFPyEnvironment(env)
# utils.validate_py_environment(env,episodes=5)