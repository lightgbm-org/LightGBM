# coding: utf-8
import numpy as np
import pytest

import lightgbm as lgb

from .utils import SERIALIZERS, pickle_and_unpickle_object


def reset_feature_fraction(boosting_round):
    return 0.6 if boosting_round < 15 else 0.8


@pytest.mark.parametrize("serializer", SERIALIZERS)
def test_early_stopping_callback_is_picklable(serializer):
    rounds = 5
    callback = lgb.early_stopping(stopping_rounds=rounds)
    callback_from_disk = pickle_and_unpickle_object(obj=callback, serializer=serializer)
    assert callback_from_disk.order == 30
    assert callback_from_disk.before_iteration is False
    assert callback.stopping_rounds == callback_from_disk.stopping_rounds
    assert callback.stopping_rounds == rounds


def test_early_stopping_callback_rejects_invalid_stopping_rounds_with_informative_errors():
    with pytest.raises(TypeError, match="early_stopping_round should be an integer. Got 'str'"):
        lgb.early_stopping(stopping_rounds="neverrrr")


@pytest.mark.parametrize("stopping_rounds", [-10, -1, 0])
def test_early_stopping_callback_accepts_non_positive_stopping_rounds(stopping_rounds):
    cb = lgb.early_stopping(stopping_rounds=stopping_rounds)
    assert cb.enabled is False


@pytest.fixture
def early_stopping_train_kwargs():
    data = np.arange(32).reshape(16, 2)
    labels = np.arange(16)
    train_set = lgb.Dataset(data, label=labels)
    return {
        "params": {
            "objective": "regression",
            "metric": "None",
            "min_data_in_leaf": 1,
            "num_leaves": 2,
            "num_threads": 1,
            "verbosity": -1,
        },
        "train_set": train_set,
        "num_boost_round": 8,
        "valid_sets": [train_set.create_valid(data, label=labels) for _ in range(2)],
        "valid_names": ["valid_0", "valid_1"],
    }


@pytest.mark.parametrize(
    ("min_delta", "first_metric_only", "best_iteration", "message"),
    [
        ([], False, 8, "Disabling min_delta for early stopping."),
        ([0.0], False, 8, "Using 0.0 as min_delta for all metrics."),
        ([0.25], False, 1, "Using 0.25 as min_delta for all metrics."),
        (0.25, False, 1, "Using 0.25 as min_delta for all metrics."),
        ([0.0, 0.25], False, 8, None),
        ([0.0, 0.75], False, 1, None),
        ([0.0, 0.75], True, 8, "Using only 0.0 as early stopping min_delta."),
    ],
)
def test_early_stopping_min_delta_broadcasting(
    min_delta, first_metric_only, best_iteration, message, early_stopping_train_kwargs, capsys
):
    evaluation_calls = 0

    def evaluate(predictions, dataset):
        nonlocal evaluation_calls
        iteration = evaluation_calls // len(early_stopping_train_kwargs["valid_sets"])
        evaluation_calls += 1
        return [("decreasing", 2.0 - iteration / 8, False), ("increasing", 3 * iteration / 8, True)]

    booster = lgb.train(
        **early_stopping_train_kwargs,
        feval=evaluate,
        callbacks=[lgb.early_stopping(2, first_metric_only=first_metric_only, min_delta=min_delta)],
    )
    assert booster.best_iteration == best_iteration
    assert set(booster.best_score) == set(early_stopping_train_kwargs["valid_names"])
    for scores in booster.best_score.values():
        assert scores == {"decreasing": 2.0 - (best_iteration - 1) / 8, "increasing": 3 * (best_iteration - 1) / 8}
    if message is not None:
        assert message in capsys.readouterr().out


@pytest.mark.parametrize(
    ("min_delta", "message"),
    [
        (-0.25, "Early stopping min_delta must be non-negative."),
        ([-0.25], "Values for early stopping min_delta must be non-negative."),
        ([0.0, -0.25], "Values for early stopping min_delta must be non-negative."),
        ([0.0, 0.0, 0.0], "Must provide a single value for min_delta or as many as metrics."),
    ],
)
def test_early_stopping_min_delta_rejects_invalid_values(min_delta, message, early_stopping_train_kwargs):
    with pytest.raises(ValueError, match=message):
        lgb.train(
            **early_stopping_train_kwargs,
            feval=lambda predictions, dataset: [("decreasing", 1.0, False), ("increasing", 1.0, True)],
            callbacks=[lgb.early_stopping(2, min_delta=min_delta)],
        )


@pytest.mark.parametrize("serializer", SERIALIZERS)
def test_log_evaluation_callback_is_picklable(serializer):
    periods = 42
    callback = lgb.log_evaluation(period=periods)
    callback_from_disk = pickle_and_unpickle_object(obj=callback, serializer=serializer)
    assert callback_from_disk.order == 10
    assert callback_from_disk.before_iteration is False
    assert callback.period == callback_from_disk.period
    assert callback.period == periods


@pytest.mark.parametrize("serializer", SERIALIZERS)
def test_record_evaluation_callback_is_picklable(serializer):
    results = {}
    callback = lgb.record_evaluation(eval_result=results)
    callback_from_disk = pickle_and_unpickle_object(obj=callback, serializer=serializer)
    assert callback_from_disk.order == 20
    assert callback_from_disk.before_iteration is False
    assert callback.eval_result == callback_from_disk.eval_result
    assert callback.eval_result is results


@pytest.mark.parametrize("serializer", SERIALIZERS)
def test_reset_parameter_callback_is_picklable(serializer):
    params = {"bagging_fraction": [0.7] * 5 + [0.6] * 5, "feature_fraction": reset_feature_fraction}
    callback = lgb.reset_parameter(**params)
    callback_from_disk = pickle_and_unpickle_object(obj=callback, serializer=serializer)
    assert callback_from_disk.order == 10
    assert callback_from_disk.before_iteration is True
    assert callback.kwargs == callback_from_disk.kwargs
    assert callback.kwargs == params
