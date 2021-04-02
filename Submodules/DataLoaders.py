import pandas as pd
import numpy as np

from sklearn.datasets import fetch_covtype, fetch_kddcup99, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical




def load_data_covtype(type='train'):
    print("Loading forest cover dataset...")
    n = 20000
    cover = fetch_covtype()
    df = pd.DataFrame(cover['data'], columns=cover['feature_names'])
    target = cover['target']
    target = target - 1
    target = to_categorical(target, 7)
    if type == 'train':
        XT, xt, YT, yt = train_test_split(np.asarray(df).astype('float32')[0:n], np.array(target)[0:n],
                                          test_size=.3)
    else:
        XT, xt, YT, yt = train_test_split(np.asarray(df).astype('float32')[n:2*n], np.array(target)[n:2*n],
                                          test_size=.3)
    return {'XT': XT.astype('float32'), 'YT': YT, 'xt': xt.astype('float32'), 'yt': yt}


def load_data_kddcup(type='train'):
    print("Loading kddcup dataset...")

    cover = fetch_kddcup99()
    df = pd.DataFrame(cover['data'], columns=cover['feature_names'])
    df['protocol_type'] = LabelEncoder().fit(list(set(df['protocol_type']))).transform(df['protocol_type'])
    df['service'] = LabelEncoder().fit(list(set(df['service']))).transform(df['service'])
    df['flag'] = LabelEncoder().fit(list(set(df['flag']))).transform(df['flag'])

    target = cover['target']
    target = [item.decode("utf-8") for item in target]
    le = LabelEncoder()
    le.fit(list(set(target)))
    target = le.transform(target)
    target = to_categorical(target, 23)

    if type == 'train':
        XT, xt, YT, yt = train_test_split(np.array(df)[:15000], np.array(target)[:15000], test_size=.3)
    else:
        XT, xt, YT, yt = train_test_split(np.array(df)[20000:40000], np.array(target)[20000:40000], test_size=.3)
    return {'XT': XT.astype('float32'), 'YT': YT, 'xt': xt.astype('float32'), 'yt': yt}


def load_data_mnist(type='train'):
    X, Y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    Y = to_categorical(Y, 10)
    if type == 'train':
        XT, xt, YT, yt = train_test_split(X[:15000], Y[:15000], test_size=.3)
    else:
        XT, xt, YT, yt = train_test_split(X[20000:40000], Y[20000:40000], test_size=.3)
    return {'XT': XT.astype('float32'), 'YT': YT, 'xt': xt.astype('float32'), 'yt': yt}