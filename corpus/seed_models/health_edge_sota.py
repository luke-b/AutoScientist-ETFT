import csv
import urllib.request

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def get_heart_failure_dataset(test_size: float = 0.2, random_state: int = 42):
    url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv" # placeholder for heart failure
    response = urllib.request.urlopen(url)
    lines = [line.decode('utf-8') for line in response.readlines()]
    reader = csv.reader(lines)
    next(reader)
    data = []
    for row in reader:
        data.append([float(x) for x in row])
    data = np.array(data)
    x = data[:, :-1]
    y = data[:, -1]

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled, y, test_size=test_size, random_state=random_state
    )
    return x_train, x_test, y_train, y_test

def main():
    x_train, x_test, y_train, y_test = get_heart_failure_dataset()
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    import pickle
    model_size_bytes = len(pickle.dumps(clf))
    fitness = acc - (max(0, model_size_bytes - 256 * 1024) / (1024 * 1024))
    print(f'METRIC: accuracy={acc}')
    print(f'METRIC: model_size_bytes={model_size_bytes}')
    print(f'METRIC: fitness={fitness}')

if __name__ == '__main__':
    main()
