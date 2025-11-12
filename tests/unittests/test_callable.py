import numpy as np
import oxonfair
from oxonfair.utils import group_metrics as gm
from matplotlib import pyplot as plt

val = np.random.randn(1000, 2)  #
val_target = np.random.randn(1000) > 0
val_groups = np.random.randn(1000) > 0

test = np.random.randn(1001, 2)
test_target = np.random.randn(1001) > 0
test_groups = np.random.randn(1001) > 0


def sigmoid(array):
    return np.stack((1 / (1 + np.exp(array[:, 0])), 1 / (1 + np.exp(-array[:, 0]))), 1)


def square_align(array):
    return np.stack((array[:, 1], 0.5 - array[:, 1]), 1)


val_dict = {"data": val, "target": val_target, "groups": val_groups}
test_dict = {"data": test, "target": test_target, "groups": test_groups}


def test_runs(use_fast=True):
    fpred = oxonfair.FairPredictor(sigmoid, val_dict, val_groups, inferred_groups=square_align,
                                   use_fast=use_fast)
    fpred.fit(gm.accuracy, gm.equal_opportunity, 0.005)
    tmp = np.asarray(fpred.evaluate(metrics={'eo': gm.equal_opportunity}))[0, 1]
    assert tmp < 0.005
    fpred.plot_frontier()
    fpred.plot_frontier(test_dict)
    fpred.evaluate()
    fpred.evaluate(test_dict)
    fpred.evaluate_fairness()
    fpred.evaluate_fairness(test_dict)
    fpred.evaluate_groups()
    fpred.evaluate_groups(test_dict)


def test_runs_slow():
    test_runs(False)


def test_runs_hybrid():
    test_runs('hybrid')


def test_fairdeep(use_fast=True, use_true_groups=False):
    fpred = oxonfair.DeepFairPredictor(val_target, val, val_groups, use_fast=use_fast, use_actual_groups=use_true_groups)
    fpred.fit(gm.accuracy, gm.equal_opportunity, 0.01)
    tmp = np.asarray(fpred.evaluate(metrics={'eo': gm.equal_opportunity}))[0, 1]
    assert tmp < 0.01
    fpred.plot_frontier()
    plt.close()
    fpred.plot_frontier(test_dict)
    plt.close()
    fpred.evaluate()
    fpred.evaluate(test_dict)
    fpred.evaluate_fairness()
    fpred.evaluate_fairness(test_dict)
    fpred.evaluate_groups()
    fpred.evaluate_groups(test_dict)


def test_fairdeep_true_g():
    test_fairdeep(True, True)


def test_fairdeep_single():
    test_fairdeep(True, 'single_threshold')


def test_fairdeep_slow(use_true_groups=False):
    test_fairdeep(False, use_true_groups)


def test_fairdeep_slow_true_g(use_true_groups=False):
    test_fairdeep_slow(use_true_groups)


def test_fairdeep_slow_single():
    test_fairdeep(False, 'single_threshold')


def test_fairdeep_hybrid(use_true_groups=False):
    test_fairdeep('hybrid', use_true_groups)


def test_fairdeep_hybrid_true_g():
    test_fairdeep('hybrid', True)


def test_fairdeep_hybrid_single():
    test_fairdeep('hybrid', 'single_threshold')
