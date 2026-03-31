# coding: utf-8
"""Tests for Metal GPU backend on Apple Silicon."""

import platform
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, make_classification, make_regression
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import train_test_split

import lightgbm as lgb
from lightgbm.libpath import _find_lib_path


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


class _LogCollector:
    def __init__(self):
        self.messages = []

    def info(self, msg):
        self.messages.append(msg)

    def warning(self, msg):
        self.messages.append(msg)


@pytest.fixture(autouse=True)
def check_metal():
    _skip_if_not_metal()


class TestMetalBasic:
    """Basic Metal functionality tests."""

    def test_packaged_metallib_lookup_outside_repo(self, monkeypatch, tmp_path):
        lib_path = Path(_find_lib_path()[0]).resolve()
        metallib_path = lib_path.parent / "default.metallib"
        if not metallib_path.is_file():
            pytest.skip("No pre-compiled default.metallib (Metal Toolchain not installed)")

        logger = _LogCollector()
        lgb.register_logger(logger)
        monkeypatch.chdir(tmp_path)

        X, y = make_classification(n_samples=100, n_features=6, random_state=42)
        data = lgb.Dataset(X, label=y)
        params = {"objective": "binary", "device": "metal", "num_leaves": 7, "verbose": 1}
        model = lgb.train(params, data, num_boost_round=5)

        assert model is not None
        joined_logs = "\n".join(logger.messages)
        assert f"Loaded pre-compiled Metal library from {metallib_path}" in joined_logs
        assert "compiling metal kernels from source" not in joined_logs.lower()

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


class TestMetalSupportedRoutes:
    """Smoke-test broader training routes advertised by the Metal backend."""

    def test_linear_tree(self):
        x = np.arange(0, 100, 0.1)
        rng = np.random.default_rng(0)
        y = 2 * x + rng.normal(0, 0.1, len(x))
        x = x[:, np.newaxis]

        params = {
            "objective": "regression",
            "metric": "l2",
            "seed": 0,
            "num_leaves": 2,
            "verbose": -1,
            "device": "metal",
        }
        base_model = lgb.train(params, lgb.Dataset(x, label=y), num_boost_round=10)
        linear_model = lgb.train(dict(params, linear_tree=True), lgb.Dataset(x, label=y), num_boost_round=10)

        base_mse = mean_squared_error(y, base_model.predict(x))
        linear_mse = mean_squared_error(y, linear_model.predict(x))
        assert linear_mse < base_mse

    def test_refit(self):
        X, y = load_breast_cancer(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbose": -1,
            "min_data": 10,
            "device": "metal",
        }
        train_set = lgb.Dataset(X_train, y_train)
        gbm = lgb.train(params, train_set, num_boost_round=20)
        err_pred = log_loss(y_test, gbm.predict(X_test))

        new_gbm = gbm.refit(X_test, y_test)
        new_err_pred = log_loss(y_test, new_gbm.predict(X_test))

        assert err_pred > new_err_pred

    def test_reset_training_data_via_update(self):
        X, y = load_breast_cancer(return_X_y=True)
        X_train, X_update, y_train, y_update = train_test_split(X, y, test_size=0.2, random_state=24)

        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbose": -1,
            "num_leaves": 15,
            "min_data": 10,
            "device": "metal",
            "seed": 0,
        }
        train_initial = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
        train_update = lgb.Dataset(X_update, label=y_update, reference=train_initial, free_raw_data=False)

        booster = lgb.Booster(params=params, train_set=train_initial)
        for _ in range(10):
            booster.update()
        loss_before = log_loss(y_update, booster.predict(X_update))

        for _ in range(10):
            booster.update(train_set=train_update)
        loss_after = log_loss(y_update, booster.predict(X_update))

        assert booster.train_set is train_update
        assert loss_after < loss_before


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
