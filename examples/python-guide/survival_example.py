# coding: utf-8
import numpy as np
from sklearn.model_selection import train_test_split

import lightgbm as lgb


def load_survival():
    """Generate synthetic survival data with signed-time label convention."""
    n = 500
    p = 5
    censoring_rate = 0.3
    rng = np.random.RandomState(seed=42)
    X = rng.randn(n, p)
    log_hazard = X[:, 0] + 0.1 * X[:, 1]
    times = rng.exponential(np.exp(-log_hazard))
    censor_times = rng.exponential(np.median(times) / censoring_rate, n)
    observed = times <= censor_times
    y = np.where(observed, np.minimum(times, censor_times), -censor_times)
    return X.astype(np.float64), y.astype(np.float64)


X, y = load_survival()

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
