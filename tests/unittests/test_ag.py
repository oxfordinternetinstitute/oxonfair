"""Tests for FairPredictor"""
import numpy as np
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.core import metrics
import oxonfair as fair
from oxonfair import FairPredictor
from oxonfair.utils import group_metrics as gm
train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')[::500]
test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
predictor = TabularPredictor(label='class').fit(train_data=train_data,time_limit=3)
new_test = test_data[~test_data['race'].isin([' Other', ' Asian-Pac-Islander', ])]  # drop other
        


def test_base_functionality():
    "not calling fit should not alter predict or predict_proba"
    fpredictor = FairPredictor(predictor, train_data, 'sex')
    fpredictor.evaluate()
    fpredictor.evaluate_fairness()
    fpredictor.evaluate_groups()
    assert (fpredictor.predict_proba(train_data) == predictor.predict_proba(train_data)).all().all()
    assert (fpredictor.predict(train_data) == predictor.predict(train_data)).all().all()
    fpredictor.evaluate(verbose=True)
    fpredictor.evaluate_fairness(verbose=True)
    fpredictor.evaluate_groups(verbose=True)
    fpredictor.evaluate_groups(verbose=True, return_original=True)
    fpredictor.evaluate_groups(return_original=True)


def test_no_groups(use_fast=True):
    "check pathway works with no groups"
    fairp = fair.FairPredictor(predictor, train_data, use_fast=use_fast)
    fairp.evaluate()
    fairp.evaluate_groups()
    fairp.evaluate_fairness()
    assert (fairp.predict_proba(train_data) == predictor.predict_proba(train_data)).all().all()
    assert (fairp.predict(train_data) == predictor.predict(train_data)).all().all()
    fairp.fit(gm.accuracy, gm.f1, 0)
    fairp.plot_frontier()
    fairp.evaluate(test_data)
    fairp.evaluate_fairness(test_data)
    fairp.evaluate_groups(test_data)
    fairp.plot_frontier(test_data)



def test_predict(use_fast=True):
    "check that fairpredictor returns the same as a standard predictor before fit is called"
    fpredictor = fair.FairPredictor(predictor, test_data, groups='sex', use_fast=use_fast)
    assert all(predictor.predict(test_data) == fpredictor.predict(test_data))
    assert all(predictor.predict_proba(test_data) == fpredictor.predict_proba(test_data))


def test_pathologoical():
    "Returns a single constant classifier"
    fpredictor = fair.FairPredictor(predictor, train_data, groups='sex', use_fast=False)
    fpredictor.fit(metrics.roc_auc, gm.equalized_odds, 0.75)
    fpredictor.plot_frontier()
    fpredictor.evaluate_fairness()


def test_pathologoical2(use_fast=True):
    "pass it the same objective twice"
    fpredictor = fair.FairPredictor(predictor, train_data, groups='sex', use_fast=use_fast)
    fpredictor.fit(gm.balanced_accuracy, gm.balanced_accuracy, 0)
    fpredictor.plot_frontier()
    fpredictor.evaluate_fairness()


def test_recall_diff(use_fast=True):
    """ Maximize accuracy while enforcing weak equalized odds,
    such that the difference in recall between groups is less than 2.5%
    This also tests the sign functionality on constraints and the objective"""

    fpredictor = fair.FairPredictor(predictor, test_data, 'sex', use_fast=use_fast)

    fpredictor.fit(gm.accuracy, gm.recall.diff, 0.025)

    # Evaluate the change in fairness (recall difference corresponds to EO)
    measures = fpredictor.evaluate_fairness()

    assert measures['original']['recall.diff'] > 0.025

    assert measures['updated']['recall.diff'] < 0.025
    measures=fpredictor.evaluate()
    acc=measures['updated']['accuracy']
    fpredictor.fit(gm.accuracy, gm.recall.diff, 0.025, greater_is_better_const=True)
    measures = fpredictor.evaluate_fairness()
    assert measures['original']['recall.diff'] > 0.025

    fpredictor.fit(gm.accuracy, gm.recall.diff, 0.025,greater_is_better_obj=False)
    assert acc>fpredictor.evaluate()['updated']['accuracy']


def test_subset(use_fast=True):
    "set up new fair class using 'race' as the protected group and evaluate on test data"
    fpredictor = fair.FairPredictor(predictor, test_data, 'race', use_fast=use_fast)

    full_group_metrics = fpredictor.evaluate_groups()
    fpredictor = fair.FairPredictor(predictor, new_test, 'race', use_fast=use_fast)
    partial_group_metrics = fpredictor.evaluate_groups()

    # Check that metrics computed over a subset of the data is consistent with metrics over all data
    for group in (' White', ' Black', ' Amer-Indian-Eskimo'):
        assert all(full_group_metrics.loc[group] == partial_group_metrics.loc[group])

    assert all(full_group_metrics.loc['Maximum difference'] >= partial_group_metrics.loc['Maximum difference'])


def test_disp_impact(use_fast=True):
    "Enforce the 4/5 rule that the max ratio between the proportion of positive decisions is less than 0.8"
    fpredictor = fair.FairPredictor(predictor, new_test, 'race', use_fast=use_fast)
    fpredictor.fit(gm.accuracy, gm.disparate_impact, 0.8)

    measures = fpredictor.evaluate_fairness()

    assert measures['original']['disparate_impact'] < 0.8

    assert measures['updated']['disparate_impact'] > 0.8


def test_min_recall(use_fast=True):
    "check that we can force recall >0.5 for all groups"
    fpredictor = fair.FairPredictor(predictor, new_test, 'race', use_fast=use_fast)
    # Enforce that every group has a recall over 0.5
    fpredictor.fit(gm.accuracy, gm.recall.min, 0.5)
    scores = fpredictor.evaluate_groups()
    assert all(scores['recall'][:-1] > 0.5)


def test_no_groups_slow():
    test_no_groups(False)

def test_predict_slow():
    test_predict(False)

def test_pathologoical2_slow():
    test_pathologoical2(False)

def test_recall_diff_slow():
    test_recall_diff(False)

def test_subset_slow():
    test_subset(False)

def test_min_recall_slow():
    test_min_recall(False)

def test_recall_diff_inferred(use_fast=True):
    "use infered attributes instead of provided attributes"
    # train two new classifiers one to predict class without using sex and one to fpredict sex without using class
    predictor, protected = fair.learners.inferred_attribute_builder(train_data, 'class', 'sex',time_limit=3)
    # Build fair object using this and evaluate fairness n.b. classifier
    # accuracy decreases due to lack of access to the protected attribute, but
    # otherwise code is doing the same thing
    fpredictor = fair.FairPredictor(predictor, train_data, 'sex', inferred_groups=protected, use_fast=use_fast)

    # Enforce that the new classifier will satisfy equalised odds (recall
    # difference between protected attributes of less than 2.5%) despite not
    # using sex at run-time

    fpredictor.fit(gm.accuracy, gm.recall.diff, 0.025)

    measures = fpredictor.evaluate_fairness()

    assert measures['original']['recall.diff'] > 0.025

    assert measures['updated']['recall.diff'] < 0.025

    # Prove that sex isn't being used by dropping it and reevaluating.

    new_data = test_data.drop('sex', axis=1, inplace=False)
    fpredictor.evaluate_groups(new_data, test_data['sex'])
    # No test needed, code just has to run with sex dropped

def test_recall_diff_inferred_slow():
    test_recall_diff_inferred(False)