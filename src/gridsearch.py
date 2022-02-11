from random import random
from tkinter.tix import COLUMN
import pandas as pd
import numpy as np

from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection

print(11)


if __name__ == 'main':
    df = pd.read_csv('../train.csv')

    X = df.drop(columns='price_range').values
    y = df.price_range.values

    classifier = ensemble.RandomForestClassifier(n_jobs=-1)
    param_grid = {
        "n_estimators": [100, 200, 300, 400],
        "max_depth": [1, 3, 5, 7],
        "criterian": ['gini', 'entropy'],
    }

    model = model_selection.GridSearchCV(
        estimator=classifier,
        param_grid=param_grid,
        scoring='accuracy',
        verbose=10,
        n_jobs=1,
        cv=5
    )

    model.fit(X, y)
    print(1)
