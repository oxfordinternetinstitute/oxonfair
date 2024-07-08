import pandas as pd
import numpy as np
import oxonfair as fair
from sklearn.ensemble import RandomForestClassifier
from oxonfair.utils import group_metrics as gm
from oxonfair.utils import conditional_group_metrics as cgm

try:
    from autogluon.tabular import TabularDataset, TabularPredictor
    AUTOGLUON_EXISTS = True
    train_data = TabularDataset("https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv")
    test_data = TabularDataset("https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv")
    predictor = TabularPredictor(label="class").fit(train_data=train_data, time_limit=3)
    new_test = test_data[~test_data["race"].isin([" Other", " Asian-Pac-Islander"])]
except ModuleNotFoundError:
    AUTOGLUON_EXISTS = False
# drop other

train_d, val_d, test_d = fair.dataset_loader.adult()
forest = RandomForestClassifier().fit(y=train_d['target'], X=train_d['data'])


def test_conditional_metrics():
    """Check that conditional metrics with weight 1 are the same as unweighted variants"""
    array1 = np.random.randint(0, 2, 100)
    array2 = np.random.randint(0, 2, 100)
    array3 = np.random.randint(0, 4, 100)
    array4 = np.random.randint(0, 5, 100)
    array5 = np.ones(100)
    cpdr = gm.pos_pred_rate.clone("cpdr", cgm.constant)
    assert np.isclose(
        gm.pos_pred_rate(array1, array2, array3), cpdr(array1, array2, array3, array4)
    )
    assert cgm.reweight_by_factor_postives(1, 1, 1, 1) == 1
    assert cgm.reweight_by_factor_postives(1, 0, 1, 1) == 1
    assert cgm.reweight_by_factor_postives(1, 0, 0, 1) == 0

    assert np.isclose(
        gm.pos_pred_rate(array1, array2, array3),
        cgm.pos_pred_rate(array1, array2, array3, array5),
    )
    assert not np.isclose(
        cgm.pos_pred_rate(array1, array2, array3, array4),
        cpdr(array1, array2, array3, array4),
    )


def make_rows(department, gender, accept, count):
    dep = np.empty(count, dtype=str)
    dep[:] = department
    gen = np.empty_like(dep)
    gen[:] = gender
    acc = np.empty(count)
    acc[:] = accept
    pred = np.empty_like(acc)
    pred[:] = np.random.binomial(1, 0.5)
    d = {"Department": dep, "Gender": gen, "Accept": acc, "Prediction": pred}
    return pd.DataFrame(d)


def test_conditional_is_consistent():
    "Check that the numbers match those reported in statistics"
    condensed = pd.read_csv("tests/Berkeley.tsv", sep="\t")
    collect = list()
    for idx in range(6):
        collect.append(
            make_rows(
                condensed["Department"][idx], "Male", 1, condensed["MaleYes"][idx]
            )
        )
        collect.append(
            make_rows(
                condensed["Department"][idx], "Female", 1, condensed["FemaleYes"][idx]
            )
        )
        collect.append(
            make_rows(
                condensed["Department"][idx], "Female", 0, condensed["FemaleNo"][idx]
            )
        )
        collect.append(
            make_rows(condensed["Department"][idx], "Male", 0, condensed["MaleNo"][idx])
        )

    complete = pd.concat(collect)
    pdr = gm.pos_pred_rate.per_group(
        complete["Prediction"], complete["Accept"], complete["Gender"]
    )
    assert np.isclose(
        pdr[0, 0], 0.30, atol=0.01
    )  # discrepency in original book vs data
    assert np.isclose(pdr[0, 1], 0.44, atol=0.01)
    cpdr = cgm.pos_pred_rate.per_group(
        complete["Prediction"],
        complete["Accept"],
        complete["Gender"],
        complete["Department"],
    )
    assert np.isclose(cpdr[0, 0], 0.43, atol=0.01)
    assert np.isclose(cpdr[0, 1], 0.38, atol=0.01)


def test_class(use_fast=True):
    if not AUTOGLUON_EXISTS:
        return
    "check base functionality is there"
    fpredictor = fair.FairPredictor(predictor, test_data,
                                    "sex", conditioning_factor="race", use_fast=use_fast)
    fpredictor.fit(gm.accuracy, gm.demographic_parity, 0.02)
    fpredictor.fit(gm.balanced_accuracy, cgm.pos_pred_rate.diff, 0.02)
    fpredictor.plot_frontier()
    fpredictor.evaluate_fairness()
    score = fpredictor.evaluate_fairness(metrics=cgm.cond_disparities, verbose=False)
    score["updated"]["pos_pred_rate_diff"] < 0.02
    fpredictor.evaluate_groups()
    fpredictor.evaluate_groups(metrics=cgm.cond_measures)


def test_class_slow():
    "check slow pathway"
    test_class(False)


def test_class_hybrid():
    "check hybrid pathway"
    test_class('hybrid')


def test_sklearn(use_fast=True):
    fpredictor = fair.FairPredictor(forest, test_d,
                                    conditioning_factor=test_d['data']["race"], use_fast=use_fast)
    fpredictor.fit(gm.accuracy, gm.demographic_parity, 0.02)
    fpredictor.fit(gm.balanced_accuracy, cgm.pos_pred_rate.diff, 0.02)
    fpredictor.plot_frontier()
    fpredictor.evaluate_fairness()
    score = fpredictor.evaluate_fairness(metrics=cgm.cond_disparities, verbose=False)
    score["updated"]["pos_pred_rate_diff"] < 0.02
    fpredictor.evaluate_groups()
    fpredictor.evaluate_groups(metrics=cgm.cond_measures)


def test_sklearn_slow():
    "check slow pathway"
    test_sklearn(False)


def test_sklearn_hybrid():
    "check hybrid pathway"
    test_sklearn('hybrid')
