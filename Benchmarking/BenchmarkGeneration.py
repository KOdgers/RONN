

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
import pandas as pd


tf.compat.v1.enable_v2_behavior()


sys.path.append('.')
from Submodules.NetworkExamples import *
from Submodules.DataLoaders import *
from Benchmarking.BenchmarkRun import *

from inspect import getmembers, isfunction, isclass

from Submodules import NetworkExamples
from Submodules import DataLoaders
Networks = getmembers(NetworkExamples, isclass)
DataLoader = getmembers(DataLoaders, isfunction)
DataLoader = [item for item in DataLoader if item[0][:4]=='load']
Check = Networks[0][1]()
BKeys = ['DataType','ModelArch','Optimizer','Starting LR','Pass 1',
         'Pass 2','Pass 3','Pass 4','Pass 5']
BenchmarkRecord = {key: [] for key in BKeys}
for i in Networks:
    for j in DataLoader:
        temp=i[1]()

        if j[0] in temp.valid_data:

            BenchInstance = Benchmarking(model=i[1](), data=j[1])
            vals = BenchInstance.Optimize()

            BenchmarkRecord['ModelArch'].append(i[0])
            BenchmarkRecord['DataType'].append(j[0])
            BenchmarkRecord['Optimizer'].append(vals[0])
            BenchmarkRecord['Starting LR'].append(vals[1])
            for l in range(1,6):
                evals = BenchInstance.Evaluate()
                BenchmarkRecord['Pass '+str(l)].append(evals[0])

BenchDF =pd.DataFrame.from_dict(BenchmarkRecord)
BenchDF.to_hdf('BenchmarkPerformance.h5',key='V1')
