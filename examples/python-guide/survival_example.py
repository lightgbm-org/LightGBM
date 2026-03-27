# coding: utf-8
import numpy as np

import lightgbm as lgb

rng = np.random.RandomState(42)
n, p = 2000, 10
X = rng.randn(n, p)

# Survival times drawn from an exponential distribution
times = rng.exponential(scale=100.0, size=n)

# Event indicators: 1 = event, 0 = censored (~30% censored)
event = rng.binomial(1, p=0.7, size=n)

# Encode labels: positive = event, negative = censored
y = np.where(event == 1, times, -times)

n_train = int(0.8 * n)
X_train, X_val = X[:n_train], X[n_train:]
y_train, y_val = y[:n_train], y[n_train:]

lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

params = {
    "objective": "survival_cox",
    "metric": ["survival_cox_nll", "concordance_index"],
    "num_leaves": 31,
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
        lgb.early_stopping(stopping_rounds=20),
        lgb.record_evaluation(evals_result),
    ],
)

print(
    f"Validation negative partial log-likelihood: {evals_result['val']['survival_cox_nll'][gbm.best_iteration - 1]:.4f}"
)
print(f"Validation concordance index: {evals_result['val']['concordance_index'][gbm.best_iteration - 1]:.4f}")

# Predictions are log-hazard ratios (higher = more risk)
preds = gbm.predict(X_val, num_iteration=gbm.best_iteration)
print(f"\nPrediction range: [{preds.min():.3f}, {preds.max():.3f}]")
