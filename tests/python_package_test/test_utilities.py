# coding: utf-8
import logging

import numpy as np
import pytest

import lightgbm as lgb


def test_register_logger(tmp_path):
    logger = logging.getLogger("LightGBM")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(levelname)s | %(message)s")
    log_filename = tmp_path / "LightGBM_test_logger.log"
    file_handler = logging.FileHandler(log_filename, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    def dummy_metric(_, __):
        logger.debug("In dummy_metric")
        return "dummy_metric", 1, True

    lgb.register_logger(logger)

    X = np.array([[1, 2, 3], [1, 2, 4], [1, 2, 4], [1, 2, 3]], dtype=np.float32)
    y = np.array([0, 1, 1, 0])
    lgb_train = lgb.Dataset(X, y, categorical_feature=[1])
    lgb_valid = lgb.Dataset(X, y, categorical_feature=[1])  # different object for early-stopping

    eval_records = {}
    callbacks = [lgb.record_evaluation(eval_records), lgb.log_evaluation(2), lgb.early_stopping(10)]
    lgb.train(
        {"objective": "binary", "metric": ["auc", "binary_error"], "verbose": 1},
        lgb_train,
        num_boost_round=10,
        feval=dummy_metric,
        valid_sets=[lgb_valid],
        callbacks=callbacks,
    )

    lgb.plot_metric(eval_records)

    expected_log = r"""
INFO | [LightGBM] [Warning] There are no meaningful features which satisfy the provided configuration. Decreasing Dataset parameters min_data_in_bin or min_data_in_leaf and re-constructing Dataset might resolve this warning.
INFO | [LightGBM] [Info] Number of positive: 2, number of negative: 2
INFO | [LightGBM] [Info] Total Bins 0
INFO | [LightGBM] [Info] Number of data points in the train set: 4, number of used features: 0
INFO | [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.500000 -> initscore=0.000000
INFO | [LightGBM] [Warning] Stopped training because there are no more leaves that meet the split requirements
DEBUG | In dummy_metric
INFO | Training until validation scores don't improve for 10 rounds
INFO | [LightGBM] [Warning] Stopped training because there are no more leaves that meet the split requirements
DEBUG | In dummy_metric
INFO | [2]	valid_0's auc: 0.5	valid_0's binary_error: 0.5	valid_0's dummy_metric: 1
INFO | [LightGBM] [Warning] Stopped training because there are no more leaves that meet the split requirements
DEBUG | In dummy_metric
INFO | [LightGBM] [Warning] Stopped training because there are no more leaves that meet the split requirements
DEBUG | In dummy_metric
INFO | [4]	valid_0's auc: 0.5	valid_0's binary_error: 0.5	valid_0's dummy_metric: 1
INFO | [LightGBM] [Warning] Stopped training because there are no more leaves that meet the split requirements
DEBUG | In dummy_metric
INFO | [LightGBM] [Warning] Stopped training because there are no more leaves that meet the split requirements
DEBUG | In dummy_metric
INFO | [6]	valid_0's auc: 0.5	valid_0's binary_error: 0.5	valid_0's dummy_metric: 1
INFO | [LightGBM] [Warning] Stopped training because there are no more leaves that meet the split requirements
DEBUG | In dummy_metric
INFO | [LightGBM] [Warning] Stopped training because there are no more leaves that meet the split requirements
DEBUG | In dummy_metric
INFO | [8]	valid_0's auc: 0.5	valid_0's binary_error: 0.5	valid_0's dummy_metric: 1
INFO | [LightGBM] [Warning] Stopped training because there are no more leaves that meet the split requirements
DEBUG | In dummy_metric
INFO | [LightGBM] [Warning] Stopped training because there are no more leaves that meet the split requirements
DEBUG | In dummy_metric
INFO | [10]	valid_0's auc: 0.5	valid_0's binary_error: 0.5	valid_0's dummy_metric: 1
INFO | Did not meet early stopping. Best iteration is:
[1]	valid_0's auc: 0.5	valid_0's binary_error: 0.5	valid_0's dummy_metric: 1
WARNING | More than one metric available, picking one to plot.
""".strip()

    gpu_lines = [
        "INFO | [LightGBM] [Info] This is the GPU trainer",
        "INFO | [LightGBM] [Info] Using GPU Device:",
        "INFO | [LightGBM] [Info] Compiling OpenCL Kernel with 16 bins...",
        "INFO | [LightGBM] [Info] GPU programs have been built",
        "INFO | [LightGBM] [Warning] GPU acceleration is disabled because no non-trivial dense features can be found",
        "INFO | [LightGBM] [Warning] Using sparse features with CUDA is currently not supported.",
        "INFO | [LightGBM] [Warning] CUDA currently requires double precision calculations.",
        "INFO | [LightGBM] [Info] LightGBM using CUDA trainer with DP float!!",
    ]
    cuda_lines = [
        "INFO | [LightGBM] [Warning] Metric auc is not implemented in cuda version. Fall back to evaluation on CPU.",
        "INFO | [LightGBM] [Warning] Metric binary_error is not implemented in cuda version. Fall back to evaluation on CPU.",
    ]
    with open(log_filename, "rt", encoding="utf-8") as f:
        actual_log = f.read().strip()
        actual_log_wo_gpu_stuff = []
        for line in actual_log.split("\n"):
            if not any(line.startswith(gpu_or_cuda_line) for gpu_or_cuda_line in gpu_lines + cuda_lines):
                actual_log_wo_gpu_stuff.append(line)

    assert "\n".join(actual_log_wo_gpu_stuff) == expected_log


def test_register_invalid_logger():
    class LoggerWithoutInfoMethod:
        def warning(self, msg: str) -> None:
            print(msg)

    class LoggerWithoutWarningMethod:
        def info(self, msg: str) -> None:
            print(msg)

    class LoggerWithAttributeNotCallable:
        def __init__(self):
            self.info = 1
            self.warning = 2

    expected_error_message = "Logger must provide 'info' and 'warning' method"

    with pytest.raises(TypeError, match=expected_error_message):
        lgb.register_logger(LoggerWithoutInfoMethod())

    with pytest.raises(TypeError, match=expected_error_message):
        lgb.register_logger(LoggerWithoutWarningMethod())

    with pytest.raises(TypeError, match=expected_error_message):
        lgb.register_logger(LoggerWithAttributeNotCallable())


def test_register_custom_logger():
    logged_messages = []

    class CustomLogger:
        def custom_info(self, msg: str) -> None:
            logged_messages.append(msg)

        def custom_warning(self, msg: str) -> None:
            logged_messages.append(msg)

    custom_logger = CustomLogger()
    lgb.register_logger(custom_logger, info_method_name="custom_info", warning_method_name="custom_warning")

    lgb.basic._log_info("info message")
    lgb.basic._log_warning("warning message")

    expected_log = ["info message", "warning message"]
    assert logged_messages == expected_log

    logged_messages = []
    X = np.array([[1, 2, 3], [1, 2, 4], [1, 2, 4], [1, 2, 3]], dtype=np.float32)
    y = np.array([0, 1, 1, 0])
    lgb_data = lgb.Dataset(X, y, categorical_feature=[1])
    lgb.train(
        {"objective": "binary", "metric": "auc"},
        lgb_data,
        num_boost_round=10,
        valid_sets=[lgb_data],
    )
    assert logged_messages, "custom logger was not called"
