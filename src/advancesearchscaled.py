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

if __name__ == '__main__':
    df = pd.read_csv('./input/train.csv')

    X = df.drop(columns='price_range').values
    y = df.price_range.values

    scl = preprocessing.StandardScaler()
    pca = decomposition.PCA()
    rf = ensemble.RandomForestClassifier()

    classifier = pipeline.Pipeline(
        [('scaling', scl), ('pca', pca), ('rf', rf)])

    param_grid = {
        'pca__n_components': np.arange(3, 8),
        "rf__n_estimators": np.arange(100, 1500),
        "rf__max_depth": np.arange(1, 20),
        "rf__criterion": ['gini', 'entropy'],
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
