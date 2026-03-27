# coding: utf-8
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

import lightgbm as lgb

# Load FLCHAIN dataset (serum free light chain and mortality)
data = fetch_openml("flchain", version=1, as_frame=True, parser="auto")
df = data.data.copy()
time = data.target.values
event = df.pop("status").values

# Encode strings as integers
df["sex"] = (df["sex"] == "F").astype(int)
df["flc.grp"] = df["flc.grp"].astype(int)
df["mgus"] = df["mgus"].astype(int)

# Encode labels: positive = event (death), negative = censored
y = np.where(event == 1, time, -time)
X = df.values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

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
        lgb.early_stopping(stopping_rounds=20, first_metric_only=True),
        lgb.record_evaluation(evals_result),
    ],
)

loss = np.asarray(evals_result["val"]["survival_cox_nll"])
c_index = np.asarray(evals_result["val"]["concordance_index"])
print(f"Validation negative partial log-likelihood: {loss[gbm.best_iteration - 1]:.4f}")
print(f"Validation concordance index: {c_index[gbm.best_iteration - 1]:.4f}")

# Predictions are log-hazard ratios (higher = more risk)
preds = gbm.predict(X_val, num_iteration=gbm.best_iteration)
print(f"\nPrediction range: [{preds.min():.3f}, {preds.max():.3f}]")
