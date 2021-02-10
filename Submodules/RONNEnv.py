import sys

sys.path.append('.')

import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs.array_spec import BoundedArraySpec
from tf_agents.trajectories import time_step

from Submodules.SaladBar import SaladBar


class RONNEnviron1D(py_environment.PyEnvironment):

    def __init__(self, train_eval='train'):
        # super().__init__()
        self.train_eval = train_eval

        self.SaladBar = SaladBar()
        self.SaladBar.add_data(self.SaladBar.load_data_covtype(self.train_eval))
        self.SaladBar.add_model(self.SaladBar.load_test_model)
        self.max_epochs = 100
        self.SaladBar.reset_bar()

        self._observation_spec = BoundedArraySpec(shape=(len(self.SaladBar.SaladWrap.get_epoch_return_1d()),),
                                                  dtype=np.float32,
                                                  minimum=[-1000] * len(self.SaladBar.SaladWrap.get_epoch_return_1d()),
                                                  maximum=[1000] * len(self.SaladBar.SaladWrap.get_epoch_return_1d()),
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
            self._episode_ended = True
            print('Reward:', reward)
            return time_step.termination(observation=self.SaladBar.SaladWrap.get_epoch_return_1d(),
                                         reward=reward
                                         )
        else:
            self.SaladBar.last_loss = self.SaladBar.SaladWrap.loss
            return time_step.transition(observation=self.SaladBar.SaladWrap.get_epoch_return_1d(),
                                        reward=reward, discount=1.0
                                        )
        # return self._current_time_step

    # def reset(self):
    #     return self._reset()

    def _reset(self):
        """Return initial_time_step."""
        self._episode_ended = False
        self.SaladBar.reset_bar()
        return time_step.restart(observation=self.SaladBar.SaladWrap.get_epoch_return_1d())
