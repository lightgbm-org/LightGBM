# coding: utf-8
from pathlib import Path

import numpy as np

preds = [np.loadtxt(str(name)) for name in Path(__file__).absolute().parent.glob("*.pred")]
np.testing.assert_allclose(preds[0], preds[1])
