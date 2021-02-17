import unittest
import sys
sys.path.append('.')

from tf_agents.environments.utils import validate_py_environment
from Submodules.RONNEnv import RONNEnviron1D
from Submodules.NetworkExamples import *
from Submodules.DataLoaders import *


class MyTestCase(unittest.TestCase):

    def test_py_environment_validate(self):
        configdict = {}
        configdict['DataLoader']= load_data_covtype
        configdict['NetworkBuilder'] = simple_seq_model
        env = RONNEnviron1D(config=configdict,train_eval='train')
        validate_py_environment(env, episodes=5)


if __name__ == '__main__':
    unittest.main()
