"""Tests for Frontier"""

import pandas as pd
import sklearn.ensemble
import sklearn.tree
import oxonfair as fair
from oxonfair.utils import group_metrics as gm

PLT_EXISTS = True
try:
    import matplotlib.pyplot as plt
    plt.title
except ModuleNotFoundError:
    PLT_EXISTS = False

classifier_type = sklearn.ensemble.RandomForestClassifier

train_data = pd.read_csv("https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv")
test_data = pd.read_csv("https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv")

# Merge and shuffle the data
total_data = pd.concat([train_data, test_data])
y = total_data["class"] == " <=50K"
total_data = total_data.drop(columns="class")
total_data = pd.get_dummies(total_data)

train = total_data.sample(frac=0.5)
val_test = total_data.drop(train.index)
train_y = y.iloc[train.index]
val_test_y = y.drop(train_y.index)
val = val_test.sample(frac=0.4)
test = val_test.drop(val.index)
val_y = y.iloc[val.index]
test_y = val_test_y.drop(val.index)
predictor = classifier_type()
predictor.fit(train, train_y)

train_dict = {"data": train, "target": train_y}
val_dict = {"data": val, "target": val_y}
test_dict = {"data": test, "target": test_y}

val_dict_g = fair.DataDict(val_y, val, val['sex_ Female'])
test_dict_g = fair.DataDict(test_y, test, test['sex_ Female'])


def test_recall_diff(use_fast=True):
    """Sweep out the found frontier for equal opportunity and check for consistency"""

    fpredictor = fair.FairPredictor(predictor, test_dict, "sex_ Female", use_fast=use_fast)

    fpredictor.fit(gm.accuracy, gm.recall.diff, 0.025)

    # Evaluate the change in fairness (recall difference corresponds to EO)
    measures = fpredictor.evaluate_fairness(verbose=False)

    thresholds = fpredictor.frontier_thresholds()
    frontier = fpredictor.frontier_scores()
    metrics = {1: fpredictor.objective1, 2: fpredictor.objective2}
    for i in range(thresholds.shape[1]):
        if use_fast is True:
            fpredictor.set_threshold(thresholds[:, i])
        else:
            fpredictor.set_threshold(thresholds[:, :, i])
        score = frontier[:, i]
        measures = fpredictor.evaluate(metrics=metrics, verbose=False)['updated']
        assert measures[1] == score[0]
        assert measures[2] == score[1]


def test_recall_diff_slow():
    "test slow pathway"
    test_recall_diff(False)


def test_recall_diff_hybrid():
    test_recall_diff('hybrid')
