from pathlib import Path

import dask.array as da
import numpy as np
from distributed import Client, LocalCluster
from sklearn.datasets import load_svmlight_file

import lightgbm as lgb

if __name__ == "__main__":
    print("loading data")

    rank_example_dir = Path(__file__).absolute().parents[2] / "lambdarank"
    X, y = load_svmlight_file(str(rank_example_dir / "rank.train"))
    group = np.loadtxt(str(rank_example_dir / "rank.train.query"))

    print("initializing a Dask cluster")

    cluster = LocalCluster(n_workers=2)
    client = Client(cluster)

    print("created a Dask LocalCluster")

    print("distributing training data on the Dask cluster")

    # split training data into two partitions
    rows_in_part1 = int(np.sum(group[:100]))
    rows_in_part2 = X.shape[0] - rows_in_part1
    num_features = X.shape[1]

    # make this array dense because we're splitting across
    # a sparse boundary to partition the data
    X = X.toarray()

    dX = da.from_array(x=X, chunks=[(rows_in_part1, rows_in_part2), (num_features,)])
    dy = da.from_array(
        x=y,
        chunks=[
            (rows_in_part1, rows_in_part2),
        ],
    )
    dg = da.from_array(x=group, chunks=[(100, group.size - 100)])

    print("beginning training")

    dask_model = lgb.DaskLGBMRanker(n_estimators=10)
    dask_model.fit(dX, dy, group=dg)
    assert dask_model.fitted_

    print("done training")
