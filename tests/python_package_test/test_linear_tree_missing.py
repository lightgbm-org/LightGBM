# coding: utf-8
import lightgbm as lgb
import numpy as np


def _leaves(node):
    if "leaf_index" in node:
        return [node]
    return _leaves(node["left_child"]) + _leaves(node["right_child"])


def _assert_nan_rows_match_leaf_value(booster, X, tree_index):
    leaf_of_row = booster.predict(
        X, start_iteration=tree_index, num_iteration=1, pred_leaf=True
    ).ravel()
    output = booster.predict(X, start_iteration=tree_index, num_iteration=1, raw_score=True)
    structure = booster.dump_model()["tree_info"][tree_index]["tree_structure"]
    saw_nan_leaf = False
    for leaf in _leaves(structure):
        rows = leaf_of_row == leaf["leaf_index"]
        nan_rows = rows & np.isnan(X).any(axis=1)
        if not nan_rows.any():
            continue
        saw_nan_leaf = True
        np.testing.assert_allclose(output[nan_rows], leaf["leaf_value"], atol=1e-5)
    assert saw_nan_leaf


def test_linear_tree_nan_in_zero_coefficient_feature_uses_leaf_value():
    # A 0/1 flag is 0 on every complete row of the NaN side, so its slope is 0.
    # Those NaN rows still belong to the leaf and must get leaf_value.
    rng = np.random.default_rng(0)
    n = 6000
    flag = rng.integers(0, 2, n).astype(float)
    z = rng.uniform(0, 1, n)
    flag[rng.uniform(0, 1, n) < 0.3] = np.nan
    y = np.where(np.isnan(flag), 3.0, np.where(flag == 0, 2.0 * z, 10.0 + 2.0 * z))
    X = np.column_stack([flag, z])
    params = {
        "objective": "regression",
        "linear_tree": True,
        "learning_rate": 0.5,
        "min_data_in_leaf": 20,
        "num_leaves": 2,
        "verbose": -1,
        "seed": 0,
        "deterministic": True,
        "num_threads": 1,
    }
    booster = lgb.train(params, lgb.Dataset(X, y), num_boost_round=3)
    _assert_nan_rows_match_leaf_value(booster, X, tree_index=1)


def test_linear_tree_leaf_with_no_complete_row_stays_constant():
    # Column 1 is missing on every row with column 0 > 0.6, so that side has
    # no row the regression can use. It must not solve an empty system.
    rng = np.random.default_rng(1)
    n = 6000
    a = rng.uniform(0, 1, n)
    b = rng.uniform(0, 1, n)
    b[a > 0.6] = np.nan
    y = np.where(np.isnan(b), 100.0 + 50.0 * (a > 0.8), 100.0 * (b > 0.5))
    y = y + rng.normal(0, 1, n)
    X = np.column_stack([a, b])
    params = {
        "objective": "regression",
        "linear_tree": True,
        "learning_rate": 0.5,
        "min_data_in_leaf": 20,
        "num_leaves": 3,
        "verbose": -1,
        "seed": 0,
        "deterministic": True,
        "num_threads": 1,
    }
    booster = lgb.train(params, lgb.Dataset(X, y), num_boost_round=3)
    _assert_nan_rows_match_leaf_value(booster, X, tree_index=1)
