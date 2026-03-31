import copy
import io
import platform
import socket
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Generator, List

import numpy as np
import pytest
from sklearn.datasets import make_blobs, make_regression
from sklearn.metrics import accuracy_score

TESTS_DIR = Path(__file__).absolute().parent


@pytest.fixture(scope="module")
def executable(pytestconfig) -> str:
    """Returns the path to the lightgbm executable."""
    return pytestconfig.getoption("execfile")


def _find_random_open_port() -> int:
    """Find a random open port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
    return port  # noqa: RET504


def _generate_n_ports(n: int) -> Generator[int, None, None]:
    return (_find_random_open_port() for _ in range(n))


def _write_dict(d: Dict, file: io.TextIOWrapper) -> None:
    for k, v in d.items():
        file.write(f"{k} = {v}\n")


def create_data(task: str, n_samples: int = 1_000) -> np.ndarray:
    """Create the appropriate data for the task.

    The data is returned as a numpy array with the label as the first column.
    """
    if task == "binary-classification":
        centers = [[-4, -4], [4, 4]]
        X, y = make_blobs(n_samples, centers=centers, random_state=42)
    elif task == "regression":
        X, y = make_regression(n_samples, n_features=4, n_informative=2, random_state=42)
    return np.hstack([y.reshape(-1, 1), X])


class DistributedMockup:
    """Simulate distributed training."""

    default_train_config = {
        "task": "train",
        "pre_partition": True,
        "machine_list_file": TESTS_DIR / "mlist.txt",
        "tree_learner": "data",
        "force_row_wise": True,
        "verbose": 0,
        "num_boost_round": 20,
        "num_leaves": 15,
        "num_threads": 2,
    }

    default_predict_config = {
        "task": "predict",
        "data": TESTS_DIR / "train.txt",
        "input_model": TESTS_DIR / "model0.txt",
        "output_result": TESTS_DIR / "predictions.txt",
    }

    def __init__(self, executable: str):
        self.executable = executable

    def worker_train(self, i: int) -> subprocess.CompletedProcess:
        """Start the training process on the `i`-th worker."""
        config_path = TESTS_DIR / f"train{i}.conf"
        cmd = [self.executable, f"config={config_path}"]
        return subprocess.run(cmd, check=True)

    def _set_ports(self) -> None:
        """Randomly assign a port for training to each worker and save all ports to mlist.txt."""
        ports = set(_generate_n_ports(self.n_workers))
        i = 0
        max_tries = 100
        while i < max_tries and len(ports) < self.n_workers:
            n_ports_left = self.n_workers - len(ports)
            candidates = _generate_n_ports(n_ports_left)
            ports.update(candidates)
            i += 1
        if i == max_tries:
            raise RuntimeError("Unable to find non-colliding ports.")
        self.listen_ports = list(ports)
        with open(TESTS_DIR / "mlist.txt", "wt") as f:
            for port in self.listen_ports:
                f.write(f"127.0.0.1 {port}\n")

    def _write_data(self, all_data: np.ndarray, worker_datasets: List[np.ndarray]) -> None:
        """Write train.txt for prediction and one training file per worker."""
        np.savetxt(str(TESTS_DIR / "train.txt"), all_data, delimiter=",")
        for i, worker_data in enumerate(worker_datasets):
            np.savetxt(str(TESTS_DIR / f"train{i}.txt"), worker_data, delimiter=",")

    def fit(self, partitions: List[np.ndarray], train_config: Dict) -> None:
        """Run the distributed training process on a single machine.

        For each worker i:
            1. The i-th partition is saved as train{i}.txt.
            2. A random port is assigned for training.
            3. A configuration file train{i}.conf is created.
            4. The lightgbm binary is called with config=train{i}.conf in another thread.
            5. The trained model is saved as model{i}.txt. Each model file only differs in data and local_listen_port.
        The whole training set is saved as train.txt.
        """
        self.train_config = copy.deepcopy(self.default_train_config)
        self.train_config.update(train_config)
        self.n_workers = self.train_config["num_machines"]
        self._set_ports()
        all_data = np.vstack(partitions)
        if self.train_config["tree_learner"] == "feature":
            self.train_config["pre_partition"] = False
            worker_datasets = [all_data for _ in range(self.n_workers)]
        else:
            worker_datasets = partitions
        self._write_data(all_data, worker_datasets)
        self.label_ = all_data[:, 0]
        futures = []
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            for i in range(self.n_workers):
                self.write_train_config(i)
                train_future = executor.submit(self.worker_train, i)
                futures.append(train_future)
            results = [f.result() for f in futures]
        for result in results:
            if result.returncode != 0:
                raise RuntimeError("Error in training")

    def predict(self, predict_config: Dict[str, Any]) -> np.ndarray:
        """Compute the predictions using the model created in the fit step.

        predict_config is used to predict the training set train.txt
        The predictions are saved as predictions.txt and are then loaded to return them as a numpy array.
        """
        self.predict_config = copy.deepcopy(self.default_predict_config)
        self.predict_config.update(predict_config)
        config_path = TESTS_DIR / "predict.conf"
        with open(config_path, "wt") as file:
            _write_dict(self.predict_config, file)
        cmd = [self.executable, f"config={config_path}"]
        result = subprocess.run(cmd, check=True)
        if result.returncode != 0:
            raise RuntimeError("Error in prediction")
        return np.loadtxt(str(TESTS_DIR / "predictions.txt"))

    def write_train_config(self, i: int) -> None:
        """Create a file train{i}.conf with the required configuration to train.

        Each worker gets a different port and piece of the data, the rest are the
        model parameters contained in `self.config`.
        """
        with open(TESTS_DIR / f"train{i}.conf", "wt") as file:
            output_model = TESTS_DIR / f"model{i}.txt"
            data = TESTS_DIR / f"train{i}.txt"
            file.write(f"output_model = {output_model}\n")
            file.write(f"local_listen_port = {self.listen_ports[i]}\n")
            file.write(f"data = {data}\n")
            _write_dict(self.train_config, file)


@lru_cache(maxsize=None)
def _metal_cli_available(executable: str) -> bool:
    if platform.system() != "Darwin":
        return False

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        data_path = tmp_dir_path / "train.txt"
        config_path = tmp_dir_path / "train.conf"
        model_path = tmp_dir_path / "model.txt"
        np.savetxt(str(data_path), create_data(task="binary-classification", n_samples=40), delimiter=",")
        with open(config_path, "wt") as file:
            _write_dict(
                {
                    "task": "train",
                    "data": data_path,
                    "output_model": model_path,
                    "objective": "binary",
                    "device_type": "metal",
                    "num_iterations": 1,
                    "num_leaves": 3,
                    "min_data_in_leaf": 1,
                    "force_row_wise": True,
                    "verbose": -1,
                },
                file,
            )
        result = subprocess.run([executable, f"config={config_path}"], capture_output=True, text=True)

    if result.returncode == 0:
        return True

    combined_output = f"{result.stdout}\n{result.stderr}".lower()
    if "metal" in combined_output or "apple silicon" in combined_output:
        return False

    raise RuntimeError(
        "Unable to determine whether the LightGBM CLI supports Metal:\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def _skip_if_not_metal_cli(executable: str) -> None:
    if not _metal_cli_available(executable):
        pytest.skip("Metal CLI backend not available")


def test_classifier(executable):
    """Test the classification task."""
    num_machines = 2
    data = create_data(task="binary-classification")
    partitions = np.array_split(data, num_machines)
    train_params = {
        "objective": "binary",
        "num_machines": num_machines,
    }
    clf = DistributedMockup(executable)
    clf.fit(partitions, train_params)
    y_probas = clf.predict(predict_config={})
    y_pred = y_probas > 0.5
    assert accuracy_score(clf.label_, y_pred) == 1.0


def test_regressor(executable):
    """Test the regression task."""
    num_machines = 2
    data = create_data(task="regression")
    partitions = np.array_split(data, num_machines)
    train_params = {
        "objective": "regression",
        "num_machines": num_machines,
    }
    reg = DistributedMockup(executable)
    reg.fit(partitions, train_params)
    y_pred = reg.predict(predict_config={})
    np.testing.assert_allclose(y_pred, reg.label_, rtol=0.2, atol=50.0)


@pytest.mark.parametrize("tree_learner", ["data", "voting", "feature"])
def test_metal_classifier(executable, tree_learner):
    """Test Metal distributed binary classification across supported tree learners."""
    _skip_if_not_metal_cli(executable)

    num_machines = 2
    data = create_data(task="binary-classification")
    partitions = np.array_split(data, num_machines)
    train_params = {
        "objective": "binary",
        "num_machines": num_machines,
        "tree_learner": tree_learner,
        "device_type": "metal",
    }
    clf = DistributedMockup(executable)
    clf.fit(partitions, train_params)
    y_probas = clf.predict(predict_config={})
    y_pred = y_probas > 0.5
    assert accuracy_score(clf.label_, y_pred) == 1.0


@pytest.mark.parametrize("tree_learner", ["data", "voting"])
def test_metal_regressor(executable, tree_learner):
    """Test Metal distributed regression across the data-based tree learners."""
    _skip_if_not_metal_cli(executable)

    num_machines = 2
    data = create_data(task="regression")
    partitions = np.array_split(data, num_machines)
    train_params = {
        "objective": "regression",
        "num_machines": num_machines,
        "tree_learner": tree_learner,
        "device_type": "metal",
    }
    reg = DistributedMockup(executable)
    reg.fit(partitions, train_params)
    y_pred = reg.predict(predict_config={})
    np.testing.assert_allclose(y_pred, reg.label_, rtol=0.2, atol=50.0)
