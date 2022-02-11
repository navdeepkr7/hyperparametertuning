from pickletools import optimize
from random import random
from tkinter.tix import COLUMN
import pandas as pd
import numpy as np

from sklearn import ensemble
from sklearn import preprocessing
from sklearn import decomposition
from sklearn import metrics
from sklearn import model_selection
from sklearn import pipeline

from functools import partial
from skopt import space
from skopt import gp_minimize


def optmize(params, param_names, x, y):
    params = dict(zip(params, param_names))
    model = ensemble.RandomForestClassifier(**params)

    kf = model_selection.StratifiedGroupKFold(n_splits=5)
    accuracy = []
    for idx in kf.split(X=x, y=y):
        train_idx, test_idx = idx[0], idx[1]

        xtrain = x[train_idx]
        ytrain = y[train_idx]

        xtest = x[test_idx]
        ytest = y[test_idx]

        model.fit(xtrain, ytrain)

        preds = model.predict(xtest)
        foldaccuracy = metrics.accuracy_score(ytest, preds)
        accuracy.append(foldaccuracy)

    return -1.0*np.mean(accuracy)


if __name__ == '__main__':
    df = pd.read_csv('./input/train.csv')

    X = df.drop(columns='price_range').values
    y = df.price_range.values

    param_space = [
        space.Integer(3, 15, name='max_depth'),
        space.Integer(100, 600, name='n_estimators'),
        space.Categorical(['gini', 'entropy'], name='criterion'),
        space.Real(0.01, 1, prior='uniform', name='max_features'),
    ]

    param_names = [
        'max_depth',
        'n_estimators',
        'criterion',
        'max_features',
    ]

    optimization_function = partial(
        optimize,
        # params=param_space,
        param_names=param_names,
        x=X,
        y=y
    )

    result = gp_minimize(
        optimization_function,
        dimensions=param_space,
        n_calls=12,
        n_random_starts=10,
        verbose=10
    )

    print(dict(zip(param_names, result.x)))
