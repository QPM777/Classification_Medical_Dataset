#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import evaluation as eval
import implementations as imp
from preprocessing import *


def train_models(x, y):
    """
    Train multiple logistic regression models on balanced dataset

    Parameters:
    x: Input features for which training are to be made.
    y: The target variable.

    Returns:
    w_values: list of parameters for each models
    """
    w_values = []
    y = y.reshape(-1)
    all_datasets = create_balanced_datasets_no_overlap(x, y, n_datasets=9)
    w0 = np.random.random((x.shape[1], 1))
    max_iter = 500
    count = 1
    for x, y in all_datasets:
        print(f"Training model {count}")
        w, _ = imp.logistic_regression(
            y, x, initial_w=w0, max_iters=max_iter, gamma=0.15
        )
        # We reuse the parameters from the previous model to start the next one
        w0 = w
        max_iter = 100
        w_values.append(w)
        count += 1
    return w_values


def train_models_with_synt_data(x, y):
    """
    Train multiple logistic regression models
    the data set is balanced using synthetic minority class data

    Parameters:
    x: Input features for which training are to be made.
    y: The target variable.

    Returns:
    w_values: list of parameters for each models
    """
    w_values = []
    y = y.reshape(-1)
    all_datasets = create_balanced_datasets_with_synthetic_data(x, y, n_datasets=7)
    w0 = np.random.random((x.shape[1], 1))
    max_iter = 500
    count = 1
    for x, y in all_datasets:
        print(f"Training model {count}")
        w, _ = imp.logistic_regression(
            y, x, initial_w=w0, max_iters=max_iter, gamma=0.15
        )
        # We reuse the parameters from the previous model to start the next one
        w0 = w
        max_iter = 100
        w_values.append(w)
        count += 1
    return w_values


def predict_with_voting(models, X, threshold):
    """
    Make predictions using average probability voting from different logistic regression models.

    Parameters:
    models: List of tuples containing trained weights and losses for each model.
    X: Input features for which predictions are to be made.

    Returns:
    y_pred: Predicted labels (-1 or 1).
    """

    # Collect the probability predictions from all models
    probabilities = []

    for w in models:
        # Calculate predicted probabilities using sigmoid function
        pred_probs = imp.sigmoid(np.dot(X, w))
        probabilities.append(pred_probs)

    # Convert probabilities to a NumPy array for easier manipulation
    probabilities = np.array(probabilities)

    # Average probabilities across all models
    avg_probabilities = np.mean(probabilities, axis=0)

    # Convert average probabilities to labels: -1 if <0.5, 1 if >=0.5
    y_pred = np.where(avg_probabilities >= threshold, 1, -1)

    return y_pred.reshape(-1)
