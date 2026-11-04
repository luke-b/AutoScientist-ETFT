import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple

def get_heart_failure_dataset(test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads the 'mstz/heart_failure' dataset from HuggingFace, preprocesses it,
    and returns train/test splits as numpy arrays.

    Returns:
        X_train, X_test, y_train, y_test
    """
    # Load dataset
    ds = load_dataset('mstz/heart_failure')
    df = pd.DataFrame(ds['train'])

    # Separate features and target
    X = df.drop(columns=['is_dead'])
    y = df['is_dead'].values

    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test
