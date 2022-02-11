from random import random
from tkinter.tix import COLUMN
import pandas as pd
import numpy as np

from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection

if __name__ == '__main__':
    df = pd.read_csv('./input/train.csv')

    X = df.drop(columns='price_range').values
    y = df.price_range.values

    # print(X)

    classifier = ensemble.RandomForestClassifier(n_jobs=None)
    param_grid = {
        "n_estimators": np.arange(100, 1500, 100),
        "max_depth": np.arange(1, 20, 1),
        "criterion": ['gini', 'entropy'],
    }

    model = model_selection.RandomizedSearchCV(
        estimator=classifier,
        param_distributions=param_grid,
        scoring='accuracy',
        verbose=10,
        n_jobs=1,
        cv=5,
        n_iter=10
    )

    model.fit(X, y)

    print(model.best_score_)
    print(model.best_estimator_.get_params())
