# coding: utf-8
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state

import lightgbm as lgb


def make_survival(*, n_samples=500, n_features=5, random_state=0):
    """Generate synthetic survival data with signed-time label convention.

    Parameters
    ----------
    n_samples : int, optional (default=500)
        Number of samples to generate.
    n_features : int, optional (default=5)
        Number of features to generate.
    random_state : int, optional (default=0)
        Random seed.

    Returns
    -------
    X : 2-d np.ndarray of shape = [n_samples, n_features]
        Input feature matrix.
    y : 1-d np.array of shape = [n_samples]
        Survival times.
    """
    censoring_rate = 0.3
    rnd_generator = check_random_state(random_state)
    X = rnd_generator.randn(n_samples, n_features)
    log_hazard = X[:, 0] + 0.5 * X[:, 1]
    times = rnd_generator.exponential(np.exp(-log_hazard))
    censor_times = rnd_generator.exponential(np.median(times) / censoring_rate, n_samples)
    observed = times <= censor_times
    y = np.where(observed, times, -censor_times)
    return X.astype(np.float64), y.astype(np.float64)


X, y = make_survival()

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

params = {
    "objective": "survival_cox",
    "metric": ["survival_cox_nll", "concordance_index"],
    "num_leaves": 10,
    "learning_rate": 0.05,
    "verbose": 0,
}

evals_result = {}
gbm = lgb.train(
    params,
    lgb_train,
    num_boost_round=200,
    valid_sets=[lgb_val],
    valid_names=["val"],
    callbacks=[
        lgb.early_stopping(stopping_rounds=5, first_metric_only=True),
        lgb.record_evaluation(evals_result),
    ],
)

# Predictions are log-hazard ratios (higher = more risk)
preds = gbm.predict(X_val, num_iteration=gbm.best_iteration)
print(f"\nPrediction range: [{preds.min():.3f}, {preds.max():.3f}]")
