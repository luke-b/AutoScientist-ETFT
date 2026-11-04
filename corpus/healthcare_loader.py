import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def get_heart_failure_dataset(test_size: float = 0.2, random_state: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads the 'mstz/heart_failure' dataset from HuggingFace, preprocesses it,
    and returns train/test splits as numpy arrays.

    Returns:
        x_train, x_test, y_train, y_test
    """
    # Load dataset
    ds = load_dataset('mstz/heart_failure')
    df = pd.DataFrame(ds['train'])

    # Separate features and target
    x = df.drop(columns=['is_dead'])
    y = df['is_dead'].values

    # Scale numerical features
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return x_train, x_test, y_train, y_test
