# coding: utf-8
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
