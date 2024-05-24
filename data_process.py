import os
import pickle
from typing import Any, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import gzip as G


def load_pickle(f: str) -> Any:
    return pickle.load(f, encoding="latin1")


def load_mnist(P, kind='train'):
    L = os.path.join(P, '%s-labels-idx1-ubyte.gz' % kind)
    I = os.path.join(P, '%s-images-idx3-ubyte.gz' % kind)

    with G.open(L, 'rb') as lp:
        lb = np.frombuffer(lp.read(), dtype=np.uint8, offset=8)

    with G.open(I, 'rb') as ip:
        im = np.frombuffer(ip.read(), dtype=np.uint8,
                           offset=16).reshape(len(lb), 784)

    return im, lb


def get_FASHION_data(
    num_training: int = 49000,
    num_validation: int = 1000,
    num_test: int = 10000,
    normalize: bool = True,
):
    X_train, y_train = load_mnist('fashion-mnist', kind='train')
    X_test, y_test = load_mnist('fashion-mnist', kind='t10k')
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask].astype(float)
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask].astype(float)
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask].astype(float)
    y_test = y_test[mask]
    if normalize:
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }


def get_RICE_data() -> dict:
    df = pd.read_csv('./rice/riceClassification.csv')
    X = np.array(df.iloc[:, :-1])
    y = np.array(df.iloc[:, -1])
    X_train_RICE, X_test_RICE, y_train_RICE, y_test_RICE = train_test_split(
        X, y, test_size=0.2, random_state=44, shuffle=True)
    X_train_RICE, X_val_RICE, y_train_RICE, y_val_RICE = train_test_split(
        X_train_RICE, y_train_RICE, test_size=0.25, random_state=1)
    data = {
        "X_train": X_train_RICE,
        "y_train": y_train_RICE,
        "X_val": X_val_RICE,
        "y_val": y_val_RICE,
        "X_test": X_test_RICE,
        "y_test": y_test_RICE,
    }
    return data
