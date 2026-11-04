import numpy as np

from corpus.healthcare_loader import get_heart_failure_dataset


def test_get_heart_failure_dataset():
    x_train, x_test, y_train, y_test = get_heart_failure_dataset(test_size=0.2, random_state=42)

    assert isinstance(x_train, np.ndarray)
    assert isinstance(x_test, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(y_test, np.ndarray)

    assert x_train.shape[0] > 0
    assert x_test.shape[0] > 0
    assert x_train.shape[1] == x_test.shape[1]
    assert y_train.shape[0] == x_train.shape[0]
    assert y_test.shape[0] == x_test.shape[0]

    # check that target is binary
    unique_y = np.unique(y_train)
    assert len(unique_y) == 2
    assert 0 in unique_y
    assert 1 in unique_y
