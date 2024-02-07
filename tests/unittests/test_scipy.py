"""Tests for FairPredictor"""
import pandas as pd
import sklearn.tree
import oxonfair as fair
from oxonfair import FairPredictor
from oxonfair.utils import group_metrics as gm
classifier_type = sklearn.tree.DecisionTreeClassifier

train_data = pd.read_csv('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
test_data = pd.read_csv('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')

#Merge and shuffle the data
total_data = pd.concat([train_data,test_data])
y = total_data['class'] == ' <=50K'
total_data = total_data.drop(columns='class')
total_data=pd.get_dummies(total_data)

train = total_data.sample(frac=0.5)
val_test = total_data.drop(train.index)
train_y = y.iloc[train.index]
val_test_y =y.drop(train_y.index)
val = val_test.sample(frac=0.4)
test = val_test.drop(val.index)
val_y=y.iloc[val.index]
test_y=val_test_y.drop(val.index)
predictor = classifier_type()
predictor.fit(train,train_y)

val_dict={'data':val, 'target':val_y}
test_dict={'data':test,'target':test_y}

def test_base_functionality():
    "not calling fit should not alter predict or predict_proba"
    fpredictor = FairPredictor(predictor, val_dict, 'sex_ Female')
    fpredictor.evaluate(val_dict)
    fpredictor.evaluate_fairness(val_dict)
    fpredictor.evaluate_groups(val_dict)
    fpredictor.evaluate()
    fpredictor.evaluate_fairness()
    fpredictor.evaluate_groups()
    assert (fpredictor.predict_proba(val_dict) == predictor.predict_proba(val_dict['data'])).all().all()
    assert (fpredictor.predict(val_dict) == predictor.predict(val_dict['data'])).all().all()
    assert (fpredictor.predict_proba(val_dict) == fpredictor.predict_proba(val_dict['data'])).all().all()

    fpredictor.evaluate(verbose=True)
    fpredictor.evaluate_fairness(verbose=True)
    fpredictor.evaluate_groups(verbose=True)
    fpredictor.evaluate_groups(verbose=True, return_original=True)
    fpredictor.evaluate_groups(return_original=True)

    fpredictor.evaluate_groups(test_dict)

def test_implicit_groups():
    val2 = val_dict.copy()
    val2['groups'] = val_dict['data']['sex_ Female']
    test2 = test_dict.copy()
    test2['groups'] = test_dict['data']['sex_ Female']
    fpredictor = FairPredictor(predictor, val2)
    fpredictor.evaluate_groups(verbose=True)
    fpredictor.evaluate_groups(test2,verbose=True)
    fpredictor.fit(gm.accuracy,gm.accuracy_parity,0.01)
    fpredictor.plot_frontier()
    fpredictor.evaluate_groups(verbose=True)
    fpredictor.evaluate_groups(test2,verbose=True)


def test_no_groups(use_fast=True):
    "check pathway works with no groups"
    fairp = fair.FairPredictor(predictor, val_dict, use_fast=use_fast)
    fairp.evaluate()
    fairp.evaluate_groups()
    fairp.evaluate_fairness()
    assert (fairp.predict_proba(val_dict) == predictor.predict_proba(val)).all().all()
    assert (fairp.predict(val_dict) == predictor.predict(val)).all().all()
    fairp.fit(gm.accuracy, gm.f1, 0)
    fairp.plot_frontier()
    fairp.evaluate(test_dict)
    fairp.evaluate_fairness(test_dict)
    fairp.evaluate_groups(test_dict)
    fairp.plot_frontier(test_dict)


def test_predict(use_fast=True):
    "check that fairpredictor returns the same as a standard predictor before fit is called"
    fpredictor = fair.FairPredictor(predictor, test_dict, groups='sex_ Female', use_fast=use_fast)
    assert (predictor.predict(test) == fpredictor.predict(test_dict)).all()
    assert (predictor.predict_proba(test) == fpredictor.predict_proba(test_dict)).all()


def test_pathologoical2(use_fast=True):
    "pass it the same objective twice"
    fpredictor = fair.FairPredictor(predictor, val_dict, groups='sex_ Female', use_fast=use_fast)
    fpredictor.fit(gm.balanced_accuracy, gm.balanced_accuracy, 0)
    fpredictor.plot_frontier()
    fpredictor.evaluate_fairness()
    fpredictor.plot_frontier(test_dict)
    fpredictor.evaluate_fairness(test_dict)



def test_recall_diff(use_fast=True):
    """ Maximize accuracy while enforcing weak equalized odds,
    such that the difference in recall between groups is less than 2.5%
    This also tests the sign functionality on constraints and the objective"""

    fpredictor = fair.FairPredictor(predictor, test_dict, 'sex_ Female', use_fast=use_fast)

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

def test_disp_impact(use_fast=True):
    "Enforce the 4/5 rule that the worst ratio between the proportion of positive decisions is greater than 0.9"
    fpredictor = fair.FairPredictor(predictor, test_dict, 'sex_ Female', use_fast=use_fast)
    fpredictor.fit(gm.accuracy, gm.disparate_impact, 0.9)

    measures = fpredictor.evaluate_fairness()

    assert measures['original']['disparate_impact'] < 0.9

    assert measures['updated']['disparate_impact'] > 0.9


def test_min_recall(use_fast=True):
    "check that we can force recall >0.5 for all groups"
    fpredictor = fair.FairPredictor(predictor, test_dict, 'sex_ Female', use_fast=use_fast)
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


def test_min_recall_slow():
    test_min_recall(False)
"""
def test_recall_diff_inferred(use_fast=True):
    "use infered attributes instead of provided attributes"
    # train two new classifiers one to predict class without using sex and one to fpredict sex without using class
    predictor, protected = fair.learners.inferred_attribute_builder(val_dict, 'class', 'sex_ Female',time_limit=3)
    # Build fair object using this and evaluate fairness n.b. classifier
    # accuracy decreases due to lack of access to the protected attribute, but
    # otherwise code is doing the same thing
    fpredictor = fair.FairPredictor(predictor, val_dict, 'sex_ Female', inferred_groups=protected, use_fast=use_fast)

    # Enforce that the new classifier will satisfy equalised odds (recall
    # difference between protected attributes of less than 2.5%) despite not
    # using sex at run-time

    fpredictor.fit(gm.accuracy, gm.recall.diff, 0.025)

    measures = fpredictor.evaluate_fairness()

    assert measures['original']['recall.diff'] > 0.025

    assert measures['updated']['recall.diff'] < 0.025

    # Prove that sex isn't being used by dropping it and reevaluating.

    new_data = test_dict.drop('sex_ Female', axis=1, inplace=False)
    fpredictor.evaluate_groups(new_data, test_dict['sex_ Female'])
    # No test needed, code just has to run with sex dropped

def test_recall_diff_inferred_slow():
    test_recall_diff_inferred(False)
"""
