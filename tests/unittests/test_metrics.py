import numpy as np
from oxonfair.utils import group_metrics as gm
from oxonfair.utils import performance as perf


def test_metrics():
    "check that core.metrics give the same answer as group metrics"
    try:
        from autogluon.core import metrics
    except ImportError:
        return
    array1 = np.random.randint(0, 2, 100)
    array2 = np.random.randint(0, 2, 100)
    array3 = np.zeros(100)
    met_list = (metrics.accuracy,
                metrics.balanced_accuracy,
                metrics.f1,
                metrics.mcc,
                metrics.precision,
                metrics.recall)
    group_met_list = (
        gm.accuracy,
        gm.balanced_accuracy,
        gm.f1,
        gm.mcc,
        gm.precision,
        gm.recall)
    for met, group_met in zip(met_list, group_met_list):
        assert np.isclose(met(array1, array2), group_met(array1, array2, array3)[0], 1e-5)


def test_metrics_identities():
    """ validity check, make sure metrics are consistent with standard identities.
     This combined with test metrics gives coverage of everything up to the clarify metrics"""
    array1 = np.random.randint(0, 2, 100)
    array2 = np.random.randint(0, 2, 100)
    array3 = np.random.randint(0, 4, 100)
    assert np.isclose(gm.pos_data_rate(array1, array2, array3),
                      1 - gm.neg_data_rate(array1, array2, array3)).all()
    assert np.isclose(gm.pos_pred_rate(array1, array2, array3),
                      1 - gm.neg_pred_rate(array1, array2, array3)).all()
    assert np.isclose(gm.true_pos_rate(array1, array2, array3),
                      1 - gm.false_neg_rate(array1, array2, array3)).all()
    assert np.isclose(gm.true_pos_rate.diff(array1, array2, array3),
                      gm.false_neg_rate.diff(array1, array2, array3))
    assert np.isclose(gm.true_neg_rate(array1, array2, array3),
                      1 - gm.false_pos_rate(array1, array2, array3)).all()
    assert np.isclose(gm.true_neg_rate.diff(array1, array2, array3),
                      gm.false_pos_rate.diff(array1, array2, array3))
    accuracy = gm.Utility([1, 0, 0, 1], 'accuracy')
    assert np.isclose(gm.accuracy(array1, array2, array3), accuracy(array1, array2, array3)).all()
    # check that additive_metrics can be called.
    assert np.isclose(gm.equalized_odds(array1, array2, array3),
                      (gm.true_pos_rate.diff(array1, array2, array3)
                       + gm.true_neg_rate.diff(array1, array2, array3)) / 2).all()


def test_performance_measures_run():
    """ check performance runs"""
    array1 = np.random.randint(0, 2, 100)
    array2 = np.random.randint(0, 2, 100)
    array3 = np.random.randint(0, 4, 100)
    array4 = np.random.randint(0, 5, 100)
    perf.evaluate(array1, array2)
    perf.evaluate_fairness(array1, array2, array3)
    perf.evaluate_per_group(array1, array2, array3)
    perf.evaluate_fairness(array1, array2, array3, array4)
    perf.evaluate_per_group(array1, array2, array3, array4)


def test_identity_groups():
    array1 = np.random.randint(0, 2, 100)
    array2 = np.random.randint(0, 2, 100)
    array3 = np.ones(100)
    perf.evaluate(array1, array2)
    perf.evaluate_per_group(array1, array2, array3)
