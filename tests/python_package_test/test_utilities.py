# coding: utf-8
import logging

import numpy as np
import pytest

import lightgbm as lgb
from lightgbm.basic import _LIB, _DummyLeveledLogger, _log_callback_with_level


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


@pytest.fixture
def _leveled_logger_cleanup():
    """Register leveled callback and guarantee cleanup regardless of test outcome."""
    # Create a dummy logger and register it to initialize the callback infrastructure
    lgb.register_leveled_logger(_DummyLeveledLogger())

    yield  # run the test

    # teardown: unregister the leveled callback using the public API
    lgb.unregister_leveled_logger()


def test_register_leveled_logger_invalid():
    class NoDebug:
        def info(self, msg):
            pass

        def warning(self, msg):
            pass

        def error(self, msg):
            pass

    class NoInfo:
        def debug(self, msg):
            pass

        def warning(self, msg):
            pass

        def error(self, msg):
            pass

    class NoWarning:
        def debug(self, msg):
            pass

        def info(self, msg):
            pass

        def error(self, msg):
            pass

    class NoError:
        def debug(self, msg):
            pass

        def info(self, msg):
            pass

        def warning(self, msg):
            pass

    class NotCallable:
        def __init__(self):
            self.debug = self.info = self.warning = self.error = 1

    with pytest.raises(TypeError, match="Logger must provide 'debug' method"):
        lgb.register_leveled_logger(NoDebug())
    with pytest.raises(TypeError, match="Logger must provide 'info' method"):
        lgb.register_leveled_logger(NoInfo())
    with pytest.raises(TypeError, match="Logger must provide 'warning' method"):
        lgb.register_leveled_logger(NoWarning())
    with pytest.raises(TypeError, match="Logger must provide 'error' method"):
        lgb.register_leveled_logger(NoError())
    with pytest.raises(TypeError, match="Logger must provide"):
        lgb.register_leveled_logger(NotCallable())


@pytest.mark.usefixtures("_leveled_logger_cleanup")
def test_log_callback_with_level_unit():
    captured: dict = {"debug": [], "info": [], "warning": [], "error": []}

    class CapturingLogger:
        def debug(self, msg: str) -> None:
            captured["debug"].append(msg)

        def info(self, msg: str) -> None:
            captured["info"].append(msg)

        def warning(self, msg: str) -> None:
            captured["warning"].append(msg)

        def error(self, msg: str) -> None:
            captured["error"].append(msg)

    lgb.register_leveled_logger(CapturingLogger())
    _log_callback_with_level(-1, b"fatal message")  # C_API_LOG_LEVEL_FATAL
    _log_callback_with_level(0, b"warning message")  # C_API_LOG_LEVEL_WARNING
    _log_callback_with_level(1, b"info message")  # C_API_LOG_LEVEL_INFO
    _log_callback_with_level(2, b"debug message")  # C_API_LOG_LEVEL_DEBUG

    assert captured["error"] == ["fatal message"]
    assert captured["warning"] == ["warning message"]
    assert captured["info"] == ["info message"]
    assert captured["debug"] == ["debug message"]


@pytest.mark.usefixtures("_leveled_logger_cleanup")
def test_register_leveled_logger_routing():
    info_messages: list = []
    warning_messages: list = []

    class CapturingLogger:
        def debug(self, msg: str) -> None:
            pass

        def info(self, msg: str) -> None:
            info_messages.append(msg)

        def warning(self, msg: str) -> None:
            warning_messages.append(msg)

        def error(self, msg: str) -> None:
            pass

    lgb.register_leveled_logger(CapturingLogger())

    # Explicitly trigger main-thread messages to avoid flakiness if training messages
    # happen to run on OpenMP worker threads (which bypass the callback).
    _log_callback_with_level(1, b"test info message")  # C_API_LOG_LEVEL_INFO
    _log_callback_with_level(0, b"test warning message")  # C_API_LOG_LEVEL_WARNING

    X = np.array([[1, 2, 3], [1, 2, 4], [1, 2, 4], [1, 2, 3]], dtype=np.float32)
    y = np.array([0, 1, 1, 0])
    lgb.train(
        {"objective": "binary", "verbose": 1},
        lgb.Dataset(X, y, categorical_feature=[1]),
        num_boost_round=2,
    )

    assert info_messages, "No Info-level messages received"
    assert warning_messages, "No Warning-level messages received"

    # Atomic delivery: no empty or whitespace-only messages (no 3-chunk artifacts)
    assert all(m.strip() for m in info_messages), "Chunk artifact in info_messages"
    assert all(m.strip() for m in warning_messages), "Chunk artifact in warning_messages"


def test_unregister_leveled_logger():
    """Test that unregister_leveled_logger() properly cleans up and allows re-registration."""
    captured_a: list = []
    captured_b: list = []

    class LoggerA:
        def debug(self, msg: str) -> None:
            pass

        def info(self, msg: str) -> None:
            captured_a.append(msg)

        def warning(self, msg: str) -> None:
            pass

        def error(self, msg: str) -> None:
            pass

    class LoggerB:
        def debug(self, msg: str) -> None:
            pass

        def info(self, msg: str) -> None:
            captured_b.append(msg)

        def warning(self, msg: str) -> None:
            pass

        def error(self, msg: str) -> None:
            pass

    try:
        # Register A, verify thunk is set
        lgb.register_leveled_logger(LoggerA())
        assert getattr(_LIB, "callback_with_level", None) is not None, "Thunk should be set after registration"

        # Unregister — thunk should be cleared
        lgb.unregister_leveled_logger()
        assert getattr(_LIB, "callback_with_level", None) is None, "Thunk should be cleared after unregistration"

        # Idempotency — second unregister should not raise
        lgb.unregister_leveled_logger()

        # Re-register with B, verify routing
        lgb.register_leveled_logger(LoggerB())
        _log_callback_with_level(1, b"hello from B")  # C_API_LOG_LEVEL_INFO
        assert captured_b == ["hello from B"], "Logger B should receive message"
        assert captured_a == [], "Logger A should not receive message after unregister"

    finally:
        lgb.unregister_leveled_logger()


@pytest.mark.usefixtures("_leveled_logger_cleanup")
def test_fatal_through_leveled_callback():
    """Test that C++ Log::Fatal() routes through the leveled callback end-to-end."""
    captured_errors: list = []

    class CapturingLogger:
        def debug(self, msg):
            pass

        def info(self, msg):
            pass

        def warning(self, msg):
            pass

        def error(self, msg):
            captured_errors.append(msg)

    lgb.register_leveled_logger(CapturingLogger())

    with pytest.raises(lgb.basic.LightGBMError, match="number of classes"):
        lgb.Booster(model_str="not_a_valid_model")

    # Fatal message was routed through the leveled callback to logger.error()
    assert captured_errors, "No Fatal-level messages received through leveled callback"
    # Single-call delivery: no empty/whitespace chunk artifacts
    assert all(m.strip() for m in captured_errors), "Chunk artifact in error messages"
    # Leveled path sends raw message — no [LightGBM] [Fatal] prefix
    assert all("[LightGBM] [Fatal]" not in m for m in captured_errors), (
        "Leveled callback should receive raw message without prefix"
    )
