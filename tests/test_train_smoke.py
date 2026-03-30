import numpy as np

from zebra_prop.train import _normalize_ids


def test_normalize_ids_handles_numpy_scalars():
    values = [np.int64(1), np.float64(2.0), "x"]
    normalized = _normalize_ids(values)
    assert normalized == [1, 2.0, "x"]
