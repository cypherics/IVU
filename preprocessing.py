import numpy as np


def split_dataset(X, Y, testSplit=0.1, valSplit=None, shuffle=True):
    """
    Returns:
        X_train, Y_train, X_val, Y_val, X_test, Y_test
    """
    X_test, Y_test = None, None
    X_val, Y_val = None, None

    if shuffle:
        idx = np.arange(X.shape[0])
        np.random.shuffle(idx)
        np.random.shuffle(idx)
        X, Y = X[idx], Y[idx]

    if testSplit is not None:
        testSplit = int(X.shape[0] * testSplit)
        X_test, Y_test = X[:testSplit], Y[:testSplit]
        X, Y = X[testSplit:], Y[testSplit:]

    if valSplit is not None:
        valSplit = int(X.shape[0] * valSplit)
        X_val, Y_val = X[:valSplit], Y[:valSplit]
        X, Y = X[valSplit:], Y[valSplit:]

    return X, Y, X_val, Y_val, X_test, Y_test
