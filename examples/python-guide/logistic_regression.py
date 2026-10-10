# coding: utf-8
"""Comparison of `binary` and `xentropy` objectives.

BLUF: The `xentropy` objective does logistic regression and generalizes
to the case where labels are probabilistic (i.e. numbers between 0 and 1).

Details: Both `binary` and `xentropy` minimize the log loss and use
`boost_from_average = TRUE` by default. Possibly the only difference
between them with default settings is that `binary` may achieve a slight
speed improvement by assuming that the labels are binary instead of
probabilistic.
"""

import time

import numpy as np
import pandas as pd
from scipy.special import expit

import lightgbm as lgb

#################
# Simulate some binary data with a single categorical and
#   single continuous predictor
rng = np.random.default_rng(seed=0)
N = 1000
X = pd.DataFrame({"continuous": range(N), "categorical": np.repeat([0, 1, 2, 3, 4], N / 5)})
CATEGORICAL_EFFECTS = [-1, -1, -2, -2, 2]
LINEAR_TERM = np.array(
    [-0.5 + 0.01 * X["continuous"][k] + CATEGORICAL_EFFECTS[X["categorical"][k]] for k in range(X.shape[0])]
) + rng.normal(loc=0, scale=1, size=X.shape[0])
TRUE_PROB = expit(LINEAR_TERM)
Y = rng.binomial(n=1, p=TRUE_PROB, size=N)
DATA = {
    "X": X,
    "probability_labels": TRUE_PROB,
    "binary_labels": Y,
    "lgb_with_binary_labels": lgb.Dataset(X, Y),
    "lgb_with_probability_labels": lgb.Dataset(X, TRUE_PROB),
}


#################
# Set up a couple of utilities for our experiments
def log_loss(preds, labels):
    """Logarithmic loss with non-necessarily-binary labels."""
    log_likelihood = np.sum(labels * np.log(preds)) / len(preds)
    return -log_likelihood


def experiment(objective, label_type, data):
    """Measure performance of an objective.

    Parameters
    ----------
    objective : {'binary', 'xentropy'}
        Objective function.
    label_type : {'binary', 'probability'}
        Type of the label.
    data : dict
        Data for training.

    Returns
    -------
    result : dict
        Experiment summary stats.
    """
    nrounds = 5
    lgb_data = data[f"lgb_with_{label_type}_labels"]
    params = {"objective": objective, "feature_fraction": 1, "bagging_fraction": 1, "verbose": -1, "seed": 123}
    time_zero = time.time()
    gbm = lgb.train(params, lgb_data, num_boost_round=nrounds)
    y_fitted = gbm.predict(data["X"])
    y_true = data[f"{label_type}_labels"]
    duration = time.time() - time_zero
    return {"time": duration, "correlation": np.corrcoef(y_fitted, y_true)[0, 1], "logloss": log_loss(y_fitted, y_true)}


#################
# Observe the behavior of `binary` and `xentropy` objectives
print("Performance of `binary` objective with binary labels:")
print(experiment("binary", label_type="binary", data=DATA))

print("Performance of `xentropy` objective with binary labels:")
print(experiment("xentropy", label_type="binary", data=DATA))

print("Performance of `xentropy` objective with probability labels:")
print(experiment("xentropy", label_type="probability", data=DATA))

# Trying this throws an error on non-binary values of y:
#   experiment('binary', label_type='probability', DATA)

# The speed of `binary` is not drastically different than
#   `xentropy`. `xentropy` runs faster than `binary` in many cases, although
#   there are reasons to suspect that `binary` should run faster when the
#   label is an integer instead of a float
K = 10
A = [experiment("binary", label_type="binary", data=DATA)["time"] for k in range(K)]
B = [experiment("xentropy", label_type="binary", data=DATA)["time"] for k in range(K)]
print(f"Best `binary` time: {min(A)}")
print(f"Best `xentropy` time: {min(B)}")
