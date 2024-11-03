#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from train import *
import matplotlib.pyplot as plt


def compute_accuracy(y, y_pred):
    y = y.flatten()
    y_pred = y_pred.flatten()
    return np.mean(y == y_pred)


def compute_f1_score(y, y_pred):
    y = y.flatten()
    y_pred = y_pred.flatten()
    tp = np.sum(np.logical_and(y == 1, y_pred == 1))
    fp = np.sum(np.logical_and(y == -1, y_pred == 1))
    fn = np.sum(np.logical_and(y == 1, y_pred == -1))
    return tp / (tp + (fp + fn) / 2)


def train_test_split(x, y, ratio=0.8, seed=42):

    np.random.seed(seed)

    train_size = int(x.shape[0] * ratio)
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    indices_train = indices[:train_size]
    indices_test = indices[train_size:]

    x_train, y_train = x[indices_train], y[indices_train]
    x_test, y_test = x[indices_test], y[indices_test]

    return x_train, x_test, y_train, y_test
