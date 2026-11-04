import pytest
import numpy as np
from corpus.healthcare_loader import get_heart_failure_dataset

def test_get_heart_failure_dataset():
    X_train, X_test, y_train, y_test = get_heart_failure_dataset(test_size=0.2, random_state=42)

    assert isinstance(X_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(y_test, np.ndarray)

    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert X_train.shape[1] == X_test.shape[1]
    assert y_train.shape[0] == X_train.shape[0]
    assert y_test.shape[0] == X_test.shape[0]

    # check that target is binary
    unique_y = np.unique(y_train)
    assert len(unique_y) == 2
    assert 0 in unique_y
    assert 1 in unique_y
