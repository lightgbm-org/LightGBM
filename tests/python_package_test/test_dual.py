# coding: utf-8
"""Tests for dual GPU+CPU support."""

import os
import platform

import pytest
from sklearn.metrics import log_loss

import lightgbm as lgb

from .utils import load_breast_cancer


@pytest.mark.skipif(
    os.environ.get("LIGHTGBM_TEST_DUAL_CPU_GPU", "0") != "1",
    reason="Set LIGHTGBM_TEST_DUAL_CPU_GPU=1 to test using CPU and GPU training from the same package.",
)
def test_cpu_and_gpu_work():
    # If compiled appropriately, the same installation will support both GPU and CPU.
    X, y = load_breast_cancer(return_X_y=True)
    data = lgb.Dataset(X, y)

    params_cpu = {"verbosity": -1, "num_leaves": 31, "objective": "binary", "device": "cpu"}
    cpu_bst = lgb.train(params_cpu, data, num_boost_round=10)
    cpu_score = log_loss(y, cpu_bst.predict(X))

    params_gpu = params_cpu.copy()
    params_gpu["device"] = "gpu"
    # Double-precision floats are only supported on x86_64 with PoCL
    params_gpu["gpu_use_dp"] = platform.machine() == "x86_64"
    gpu_bst = lgb.train(params_gpu, data, num_boost_round=10)
    gpu_score = log_loss(y, gpu_bst.predict(X))

    rel = 1e-6 if params_gpu["gpu_use_dp"] else 1e-4
    assert cpu_score == pytest.approx(gpu_score, rel=rel)
    assert gpu_score < 0.242
