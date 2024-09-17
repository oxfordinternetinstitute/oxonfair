"""Standard evaluation functionality
Contains 3 main functions:
1. evaluate which reports performance measures over the whole set.
2. evaluate_fairness which reports fairness measures over the whole set
3. evaluate_per_group which reports performance measures per group"""

import numpy as np
import pandas as pd
from . import group_metrics
from .group_metric_classes import BaseGroupMetric
from .scipy_metrics_cont_wrapper import ScorerRequiresContPred

AUTOGLUON_EXISTS = True
try:
    from autogluon.core.metrics import Scorer
except ModuleNotFoundError:
    AUTOGLUON_EXISTS = False


def evaluate(target, prediction, *,
             metrics=None, verbose=True, threshold="auto") -> pd.DataFrame:
    """Compute standard performance measures based on target values (i.e. y, or ground-truth)
        and predictions.
    parameters
    ----------
    target: the values a predictor should be making can be either values from {0,1} or {-1,1}
    prediction: the predictions made by the model
            predictions can be either logits or probabilities or discrete values, however,
            if discrete roc will not be informative.
    metrics: (optional) a dictionary where the keys are metric names and the elements are either
            scoreables or group metrics. If not provided report the standard metrics
            reported by autogluon on binary predictors
    verbose: (optional, bool default true) if the indexes in the returned dataframe should be long names (true),
            or the short names used as dictionary keys.
    returns
    -------
    a pandas dataset containing rows indexed by measure name
    """
    if metrics is None:
        metrics = group_metrics.ag_metrics

    threshold = find_threshold(threshold, prediction)
    groups = np.ones_like(prediction)

    return evaluate_fairness(
        target,
        prediction,
        groups,
        metrics=metrics,
        verbose=verbose,
        threshold=threshold)


def evaluate_fairness(target, prediction, groups, factor=None, *,
                      metrics=None, verbose=True, threshold="auto") -> pd.DataFrame:
    """Compute standard fairness metrics.

    parameters
    ----------
    parameters
    ----------
    target: the values a predictor should be making can be either values from {0,1} or {-1,1}
    prediction: the predictions made by the model
            predictions can be either logits or probabilities or discrete values, however,
            if discrete roc will not be informative.
    groups: numpy array containing discrete group labelling
    metrics: (optional) a dictionary where the keys are metric names and the elements are either
            scoreables or group metrics. If not provided report the standard metrics
            reported by AWS clarify
    verbose: (optional, bool) if the indexes in the returned dataframe should be long names (true),
            or the short names used as dictionary keys.
    returns
    -------
    a pandas dataset containing rows indexed by fairness measure name
    """
    target = target.squeeze()
    assert np.unique(target).shape[0] <= 2, 'More than two target labels used. OxonFair only works with binary predictors'
    threshold = find_threshold(threshold, prediction)
    if groups is None:
        groups = np.ones_like(target)

    if metrics is None:
        metrics = group_metrics.default_fairness_measures

    values = np.zeros(len(metrics))
    names = []
    for i, k in enumerate(metrics.keys()):
        if verbose is False:
            names.append(k)
        else:
            names.append(metrics[k].name)
        values[i] = dispatch_metric(
            metrics[k], target, prediction, groups, factor, threshold=threshold)

    return pd.DataFrame(values, index=names)[0]


def evaluate_per_group(target, prediction, groups, factor=None, *,
                       metrics=None, threshold="auto", verbose=True):
    """Compute standard performance measures per group

    parameters
    ----------
    target: the values a predictor should be making can be either values from {0,1} or {-1,1}
    prediction: the predictions made by the model
            predictions can be either logits or probabilities or discrete values, however,
            if discrete roc will not be informative.
    groups: numpy array containing discrete group labelling
    metrics: (optional) a dictionary where the keys are metric names and the elements are either
            scoreables or group metrics. If not provided report the standard metrics
            reported by autogluon on binary predictors
    verbose: (optional, bool) if the indexes in the returned dataframe should be long names (true),
            or the short names used as dictionary keys.
    returns
    -------
    a pandas dataset containing rows indexed by fairness measure name
    """
    target = target.squeeze()
    assert np.unique(target).shape[0] <= 2, 'More than two target labels used. OxonFair only works with binary predictors'

    threshold = find_threshold(threshold, prediction)

    if metrics is None:
        metrics = group_metrics.default_group_metrics

    prediction = np.asarray(prediction)
    group_names = np.unique(groups)

    names = list(metrics.keys())

    overall_scores = list(
        map(lambda n: dispatch_metric(
                metrics[n], target, prediction, groups, factor, threshold=threshold),
            names)
        )
    scores = list(
        map(
            lambda n: dispatch_metric_per_group(
                metrics[n], target, prediction, groups, factor, threshold=threshold),
            names)
        )
    scores = np.stack(scores)
    overall_scores = np.stack(overall_scores)

    if verbose is False:
        pandas_names = names
    else:
        pandas_names = list(map(lambda n: metrics[n].name, names))
    gap = (scores.max(-1) - scores.min(-1)).reshape(-1, 1)
    collect = np.hstack((overall_scores.reshape(-1, 1), scores, gap))
    out = pd.DataFrame(
        collect.T,
        index=(["Overall"] + group_names.tolist() + ["Maximum difference"]),
        columns=pandas_names)
    if verbose:
        out.index.name = "Groups"
    else:
        out.index.name = "groups"
    return out


def dispatch_metric(metric, y_true, proba, groups, factor, *, threshold) -> float:
    """Helper function for making sure different types of Scorer and GroupMetrics get the right data

    Parameters
    ----------
    metric: a BaseGroupMetric or Scorable
    y_true: a numpy array indicating positive or negative labels
    proba: a 2xdatapoints numpy or pandas array
    groups: a numpy array indicating group membership.

    Returns
    -------
     a numpy array containing the score provided by metrics
    """

    threshold = find_threshold(threshold, proba)
    # y_true might take values other than +1,-1
    y_true = y_true > 0

    proba = np.asarray(proba)
    try:
        if isinstance(metric, BaseGroupMetric):
            if metric.cond_weights is None:
                return metric(y_true, proba > threshold, groups)[0]
            return metric(y_true, proba > threshold, groups, factor)[0]

        if isinstance(metric, ScorerRequiresContPred) or (
                AUTOGLUON_EXISTS
                and isinstance(metric, Scorer)
                and (metric.needs_pred is False)):
            return metric(y_true, proba)
        return metric(y_true, proba > threshold)
    except ValueError:
        return np.nan


def dispatch_metric_per_group(
        metric, y_true: np.ndarray, proba: np.ndarray, groups: np.ndarray, factor: np.ndarray, *,
        threshold="auto") -> np.ndarray:
    """Helper function for making sure different types of Scorer and GroupMetrics get the right data
    parameters
    ----------
    metric: a GroupMetric or Scorable
    y_true: a binary numpy array indicating positive or negative labels
    proba: a 2xdatapoints numpy or pandas array
    groups: a numpy array indicating group membership.

    returns
    -------
    a numpy array containing the per group score provided by metrics"""
    threshold = find_threshold(threshold, proba)

    # y_true might take values other than +1,-1
    y_true = y_true > 0

    if isinstance(metric, group_metrics.GroupMetric):
        if metric.cond_weights is None:
            return metric.per_group(y_true, proba > threshold, groups)[0]
        return metric.per_group(y_true, proba > threshold, groups, factor)[0]
    unique = np.unique(groups)
    out = np.empty_like(unique, dtype=float)
    if isinstance(metric, ScorerRequiresContPred) or (
            AUTOGLUON_EXISTS
            and isinstance(metric, Scorer)
            and (metric.needs_pred is False)):
        for i, grp in enumerate(unique):
            mask = grp == groups
            try:
                out[i] = metric(y_true[mask], proba[mask])
            except ValueError:
                out[i] = np.nan
    else:
        out = metric(y_true, proba > threshold, groups)

    return out


def find_threshold(threshold, predictions):
    "heleper fn to automatically decide if we are dealing with logits or sigmoids"
    if threshold == "auto":
        if predictions.min() < 0 or predictions.max() > 1.0:
            return 0
        return 0.5
    return threshold
