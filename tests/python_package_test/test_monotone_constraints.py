import lightgbm as lgb
import numpy as np
import pandas as pd


def test_monotone_constraints_method_without_constraints():
    """Using monotone_constraints_method without monotone_constraints
    should not crash and should still train successfully.
    """

    size = 60

    df = pd.DataFrame({
        "x": np.random.uniform(size=size),
        "c": pd.Categorical(np.random.choice([0, 1, 2], size=size)),
        "y": np.random.uniform(size=size),
    })

    dataset = lgb.Dataset(df[["x", "c"]], df["y"], categorical_feature="auto")

    params = {
        "monotone_constraints_method": "advanced"
    }

    # Training should run without crashing
    model = lgb.train(params, dataset)

    assert model is not None