import unittest
from Submodules.RONNEnv import RONNEnviron1D
from tf_agents.environments.utils import validate_py_environment


class MyTestCase(unittest.TestCase):

    def test_py_environment_validate(self):
        env = RONNEnviron1D(train_eval='train')
        validate_py_environment(env,episodes=5)


if __name__ == '__main__':
    unittest.main()
