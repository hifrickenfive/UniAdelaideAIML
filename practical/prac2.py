from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
# obtain the iris dataset, and add some missing values to it
X, y = load_iris(return_X_y=True)
mask = np.random.randint(0, 2, size=X.shape).astype(bool)
X[mask] = np.nan
X_train, X_test, y_train, _ = train_test_split(X, y, test_size=100, random_state=0)
