import sys

sys.path.append('.')

import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs.array_spec import BoundedArraySpec
from tf_agents.trajectories import time_step

from Submodules.SaladBar import SaladBar
from Submodules.DataLoaders import *
from Submodules.NetworkExamples import *


class RONNEnviron1D(py_environment.PyEnvironment):

    def __init__(self, train_eval='train'):
        # super().__init__()
        self.train_eval = train_eval

        self.SaladBar = SaladBar()
        self.SaladBar.add_data(load_data_mnist(self.train_eval))
        self.SaladBar.add_model(load_test_cnn_mnist)
        self.max_epochs = 100
        self.SaladBar.reset_bar()

        self._observation_spec = BoundedArraySpec(shape=(len(self.SaladBar.SaladWrap.get_epoch_return_1d()),),
                                                  dtype=np.float32,
                                                  minimum=[-1000] * len(self.SaladBar.SaladWrap.get_epoch_return_1d()),
                                                  maximum=[1000] * len(self.SaladBar.SaladWrap.get_epoch_return_1d()),
                                                  name='observation')
        #
        self._action_spec = BoundedArraySpec(shape=(len(self.SaladBar.SaladWrap.get_opt_1d()),),
                                             dtype=np.float32,
                                             name='action',
                                             minimum=[-7]*len(self.SaladBar.SaladWrap.get_opt_1d()),
                                             maximum=[0]*len(self.SaladBar.SaladWrap.get_opt_1d()))
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

    def _step(self, action):
        if self._episode_ended:
            return self.reset()
        """Apply action and return new time_step."""

        reward = self.SaladBar.advance(action)
        # print(type(reward))
        print('Epoch #',self.SaladBar.SaladWrap.epochs_run,'  Loss:',self.SaladBar.SaladWrap.loss,' Prev. Loss:', self.last_best_loss)
        # print(self.SaladBar.SaladWrap.epochs_run, self.max_epochs)

        if (self.SaladBar.SaladWrap.loss <= self.last_best_loss):
            self.bad_epochs +=1
        else:
            self.bad_epochs = 0
            self.last_best_loss =self.SaladBar.SaladWrap.loss

        if ((self.bad_epochs >= 3) or
                (self.SaladBar.SaladWrap.epochs_run == self.max_epochs)):
            self._episode_ended = True
            print('Reward:', self.SaladBar.SaladWrap.loss)
            return time_step.termination(observation=self.SaladBar.SaladWrap.get_epoch_return_1d(),
                                         reward=self.SaladBar.SaladWrap.loss
                                         )
        else:
            self.SaladBar.last_loss = self.SaladBar.SaladWrap.loss
            return time_step.transition(observation=self.SaladBar.SaladWrap.get_epoch_return_1d(),
                                        reward=0, discount=1
                                        )


    def _reset(self):
        """Return initial_time_step."""
        self._episode_ended = False
        self.bad_epochs =0
        self.SaladBar.reset_bar()
        self.last_best_loss = self.SaladBar.SaladWrap.loss
        print('Epoch #',self.SaladBar.SaladWrap.epochs_run,'  Loss:',self.SaladBar.SaladWrap.loss,' Prev. Loss:', self.SaladBar.last_loss)

        return time_step.restart(observation=self.SaladBar.SaladWrap.get_epoch_return_1d())
