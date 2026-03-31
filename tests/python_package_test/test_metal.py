# coding: utf-8
"""Tests for Metal GPU backend on Apple Silicon."""

import os
import platform

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import log_loss

import lightgbm as lgb


def _skip_if_not_metal():
    """Skip test if Metal backend is not available."""
    if platform.system() != "Darwin":
        pytest.skip("Metal backend only available on macOS")
    # Try to create a booster with Metal device — if it fails, Metal is not compiled in
    try:
        X = np.random.randn(10, 2)
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=float)
        data = lgb.Dataset(X, label=y)
        params = {"device": "metal", "verbose": -1, "num_leaves": 2, "num_iterations": 1}
        lgb.train(params, data)
    except lgb.basic.LightGBMError as e:
        if "Metal" in str(e) or "metal" in str(e):
            pytest.skip(f"Metal backend not available: {e}")
        raise


def _assert_binary_models_similar(y, cpu_preds, metal_preds, max_loss_gap=0.01):
    """Compare CPU and Metal binary classifiers using training-set quality."""
    cpu_loss = log_loss(y, cpu_preds)
    metal_loss = log_loss(y, metal_preds)
    assert cpu_loss < 0.7, f"CPU model has poor log_loss: {cpu_loss}"
    assert metal_loss < 0.7, f"Metal model has poor log_loss: {metal_loss}"
    assert abs(cpu_loss - metal_loss) < max_loss_gap, (
        f"CPU and Metal log_loss differ too much: {cpu_loss:.6f} vs {metal_loss:.6f}"
    )


@pytest.fixture(autouse=True)
def check_metal():
    _skip_if_not_metal()


class TestMetalBasic:
    """Basic Metal functionality tests."""

    def test_binary_classification(self):
        X, y = make_classification(n_samples=500, n_features=10, random_state=42)
        data = lgb.Dataset(X, label=y)

        params_cpu = {"objective": "binary", "num_leaves": 15, "verbose": -1}
        cpu_model = lgb.train(params_cpu, data, num_boost_round=20)

        data_metal = lgb.Dataset(X, label=y)
        params_metal = {**params_cpu, "device": "metal"}
        metal_model = lgb.train(params_metal, data_metal, num_boost_round=20)

        cpu_preds = cpu_model.predict(X)
        metal_preds = metal_model.predict(X)

        # FP32 accumulation allows small differences
        np.testing.assert_allclose(cpu_preds, metal_preds, rtol=1e-3, atol=1e-4)

    def test_regression(self):
        X, y = make_regression(n_samples=500, n_features=10, random_state=42)
        data = lgb.Dataset(X, label=y)

        params_cpu = {"objective": "regression", "num_leaves": 15, "verbose": -1}
        cpu_model = lgb.train(params_cpu, data, num_boost_round=20)

        data_metal = lgb.Dataset(X, label=y)
        params_metal = {**params_cpu, "device": "metal"}
        metal_model = lgb.train(params_metal, data_metal, num_boost_round=20)

        cpu_preds = cpu_model.predict(X)
        metal_preds = metal_model.predict(X)

        np.testing.assert_allclose(cpu_preds, metal_preds, rtol=1e-3, atol=1e-2)

    def test_multiclass(self):
        X, y = make_classification(
            n_samples=500, n_features=10, n_classes=3,
            n_informative=6, random_state=42
        )
        data = lgb.Dataset(X, label=y)

        params_cpu = {
            "objective": "multiclass", "num_class": 3,
            "num_leaves": 15, "verbose": -1
        }
        cpu_model = lgb.train(params_cpu, data, num_boost_round=10)

        data_metal = lgb.Dataset(X, label=y)
        params_metal = {**params_cpu, "device": "metal"}
        metal_model = lgb.train(params_metal, data_metal, num_boost_round=10)

        cpu_preds = cpu_model.predict(X)
        metal_preds = metal_model.predict(X)

        np.testing.assert_allclose(cpu_preds, metal_preds, rtol=1e-3, atol=1e-4)


class TestMetalBinVariants:
    """Test all three histogram kernel variants."""

    @pytest.mark.parametrize("max_bin", [15, 63, 255])
    def test_kernel_variants(self, max_bin):
        X, y = make_classification(n_samples=200, n_features=8, random_state=42)
        data_cpu = lgb.Dataset(X, label=y, params={"max_bin": max_bin})
        data_metal = lgb.Dataset(X, label=y, params={"max_bin": max_bin})

        params = {
            "objective": "binary", "num_leaves": 7,
            "max_bin": max_bin, "verbose": -1, "min_data_in_leaf": 1
        }
        cpu_model = lgb.train(params, data_cpu, num_boost_round=10)

        params["device"] = "metal"
        metal_model = lgb.train(params, data_metal, num_boost_round=10)

        cpu_preds = cpu_model.predict(X)
        metal_preds = metal_model.predict(X)

        np.testing.assert_allclose(cpu_preds, metal_preds, rtol=1e-3, atol=1e-4)

    @pytest.mark.parametrize("max_bin", [15, 63, 255])
    def test_kernel_variants_large_dataset(self, max_bin):
        """Exercise the multi-workgroup path for all histogram kernels."""
        X, y = make_classification(n_samples=5000, n_features=20, random_state=42)
        data_cpu = lgb.Dataset(X, label=y, params={"max_bin": max_bin})
        data_metal = lgb.Dataset(X, label=y, params={"max_bin": max_bin})

        params = {"objective": "binary", "num_leaves": 31, "max_bin": max_bin, "verbose": -1}
        cpu_model = lgb.train(params, data_cpu, num_boost_round=10)

        params["device"] = "metal"
        metal_model = lgb.train(params, data_metal, num_boost_round=10)

        _assert_binary_models_similar(y, cpu_model.predict(X), metal_model.predict(X))


class TestMetalScalability:
    """Test with various dataset sizes including larger ones."""

    @pytest.mark.parametrize("n_samples", [100, 500, 1000, 5000, 10000])
    def test_dataset_sizes(self, n_samples):
        X, y = make_classification(n_samples=n_samples, n_features=20, random_state=42)
        data_cpu = lgb.Dataset(X, label=y)
        data_metal = lgb.Dataset(X, label=y)

        params = {"objective": "binary", "num_leaves": 31, "max_bin": 255, "verbose": -1}
        cpu_model = lgb.train(params, data_cpu, num_boost_round=10)

        params["device"] = "metal"
        metal_model = lgb.train(params, data_metal, num_boost_round=10)

        cpu_preds = cpu_model.predict(X)
        metal_preds = metal_model.predict(X)

        # FP32 accumulation allows some prediction drift vs CPU's FP64, especially
        # for larger datasets and multi-workgroup reductions. Loss should remain close.
        _assert_binary_models_similar(y, cpu_preds, metal_preds)


class TestMetalOptions:
    """Test Metal-specific configuration options."""

    def test_gpu_use_dp_forced_false(self):
        """Metal should force gpu_use_dp=false and warn."""
        X, y = make_classification(n_samples=100, n_features=4, random_state=42)
        data = lgb.Dataset(X, label=y)
        params = {
            "objective": "binary", "device": "metal",
            "gpu_use_dp": True, "verbose": -1,
            "num_iterations": 1
        }
        # Should not crash — just warn and use FP32
        model = lgb.train(params, data)
        assert model is not None

    def test_constant_hessian(self):
        """Binary classification uses constant hessian — test this path."""
        X, y = make_classification(n_samples=300, n_features=5, random_state=42)
        data_cpu = lgb.Dataset(X, label=y)
        data_metal = lgb.Dataset(X, label=y)

        params = {"objective": "binary", "num_leaves": 7, "verbose": -1}
        cpu_model = lgb.train(params, data_cpu, num_boost_round=10)

        params["device"] = "metal"
        metal_model = lgb.train(params, data_metal, num_boost_round=10)

        np.testing.assert_allclose(
            cpu_model.predict(X), metal_model.predict(X), rtol=1e-3, atol=1e-4
        )

    def test_bagging(self):
        """Test with bagging enabled."""
        X, y = make_classification(n_samples=500, n_features=10, random_state=42)
        data_cpu = lgb.Dataset(X, label=y)
        data_metal = lgb.Dataset(X, label=y)

        params = {
            "objective": "binary", "num_leaves": 15, "verbose": -1,
            "bagging_fraction": 0.8, "bagging_freq": 1, "seed": 42
        }
        cpu_model = lgb.train(params, data_cpu, num_boost_round=10)

        params["device"] = "metal"
        metal_model = lgb.train(params, data_metal, num_boost_round=10)

        np.testing.assert_allclose(
            cpu_model.predict(X), metal_model.predict(X), rtol=1e-2, atol=1e-3
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
