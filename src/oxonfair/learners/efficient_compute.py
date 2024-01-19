"""Implements efficient methods for fast computation of binary metrics"""
import logging
from typing import Callable, Tuple, List
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
logger = logging.getLogger(__name__)


def compute_metric(
        metric: Callable,
        y_true: np.ndarray,
        proba: np.ndarray,
        groups: np.ndarray,
        group_prediction: np.ndarray,
        weights: np.ndarray) -> np.ndarray:
    """takes probability scores, and offsets them according to weights[group_prediction].
        then selects the max and computes a fairness metric

    parameters
    ----------
    metric: a BaseGroupMetric
    y_true: a numpy array containing the target labels
    proba: a numpy array containing the soft classifier responses.
    groups: a numpy array containing group assignment
    threshold assignment: a numpy array containing group predictions, when groups are infered
        this differs from groups
    weights: a numpy array containing the set of per group offsets
    returns
    -------
    a numpy array of scores for each choice of weight
    """

    score = np.zeros((weights.shape[-1]))
    y_true = np.asarray(y_true)
    group_prediction = group_prediction.astype(int)
    for i in range(weights.shape[-1]):
        proba_update = proba.copy()
        proba_update[:, 1] += weights[group_prediction, i]
        pred = proba_update.argmax(-1)
        score[i] = metric(y_true, pred, groups)
    return score


def keep_front(solutions: np.ndarray, initial_weights: np.ndarray, directions: np.ndarray,
               *, tol=1e-12) -> Tuple[np.ndarray, np.ndarray]:
    """Modified from
        https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
        Returns Pareto efficient row subset of solutions and its associated weights
        Direction if a vector that governs if the frontier should maximize or minimize each
        direction.
        Where an element of direction is positive frontier maximizes, negative, it mimizes.
        parameters
        ----------
        solutions: a numpy array of values that are evaluated to find the frontier
        initial_weights: a numpy array of corresponding weights
        directions: a binary vector containing [+1,-1] indicating if greater or lower solutions are
            prefered
        tol: a float indicating if points that are almost dominated (i.e. they are within tol of
            another point in the frontier)  should be dropped.
            This is used to eliminate ties, and to discard most of the constant classifiers.
        returns
        -------
        a pair of numpy arrays.
            1. reduced set of solutions associated with the Pareto front
            2. reduced set of weights associated with the Pareto front
    """

    front = solutions.T.copy()
    weights = initial_weights.T.copy()
    front *= directions
    # drop all Nans
    mask = np.logical_not(np.isnan(front).any(1))
    front = front[mask]
    weights = weights[mask]
    # drop all points worse than the extrema of the front
    mask = np.greater_equal(front[:, 1], front[front[:, 0].argmax(), 1])
    mask *= np.greater_equal(front[:, 0], front[front[:, 1].argmax(), 0])
    front = front[mask]
    weights = weights[mask]
    # sort points by decreasing sum of coordinates
    # Add 10**-8 * magnitude of w so that in the event of a near tie, pick points close to
    # the mean first
    mean = weights.mean(0)
    modifier = -(10**-8) * np.abs(weights - mean).sum(1)
    order = (front.sum(1) + modifier).argsort()[::-1]
    front = front[order]
    weights = weights[order]
    # initialize a boolean mask for currently undominated points
    undominated = np.ones(front.shape[0], dtype=bool)

    for i in range(front.shape[0]):
        size = front.shape[0]
        # process each point in turn
        if i >= size:
            break
        # find all points not dominated by i
        # since points are sorted by coordinate sum
        # i cannot dominate any points in 1,...,i-1
        undominated[i] = True  # Bug fix missing from online version
        undominated[i + 1:size] = (front[i + 1:size] >= front[i] + tol).any(1)
        front = front[undominated[:size]]
        weights = weights[undominated[:size]]

    weights = weights.T
    front *= directions
    front = front.T
    order = (front[0]).argsort()

    return front[:, order], weights[:, order]

def build_grid_inner(accum_count, mesh, groups):
    """Sample from accum_count using mesh"""
    acc = accum_count[0][mesh[0]]
    for i in range(1, len(accum_count)):
        acc = acc + accum_count[i][mesh[i]]  # variable matrix size mean += doesn't work
    assert acc.shape[-2:] == (4, groups)
    acc = acc.reshape(-1, 4, groups)
    acc = acc.transpose(1, 0, 2)
 
    return acc

def build_grid(accum_count: np.ndarray, bottom, top, metric1: Callable,
               metric2: Callable, *, steps=25) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    """Part of efficient grid search.
    This uses the fact that metrics can be computed efficiently as a function of TP,FP,FN and TN.
    By sorting the data per assigned group  we can efficiently compute these four values by looking
    at the cumlative sum of positive and negative labelled data (provided by ordered encode).
    This brings any subsequent computation of metrics down to O(1) in the dataset size.
    Parameters
    ----------
    accum_count:
    bottom: a single number or per group numpy array indicating where the grid should start
    top: a single number or per group numpy array indicating where the grid should stop
    metric1: a BaseGroupMetric
    metric2: a BaseGroupMetric
    steps: (optional) The number of divisions per group
    returns
    -------
    a tupple of three numpy arrays:
        1. the scores of metric1 and metric 2 computed for each choice of threshold
        2. the indicies corresponding to thresholds
        3. the step offset used.
    """
    #assigned_groups = len(bottom)
    groups = accum_count[0].shape[-1]
    step = [(t - b) / steps for t, b in zip(top, bottom)]

    mesh_indices = [np.unique(np.floor(np.arange(np.floor(b), np.ceil(t + 1),
                                                 max(1, s))).astype(int))
                    for b, t, s in zip(bottom, top, step)]
    mesh = np.meshgrid(*mesh_indices, sparse=True)

    acc=build_grid_inner(accum_count, mesh, groups)
    met1 = metric1(acc)
    assert len(met1.shape) == 1
    met2 = metric2(acc)
    assert len(met2.shape) == 1
    score = np.stack((met1, met2), 0)
    return score, mesh_indices, np.maximum(1, np.asarray(step))

def build_grid2(accum_count1: np.ndarray, accum_count2: np.ndarray, bottom, top, metric1: Callable,
               metric2: Callable, *, steps=25) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    """Part of efficient grid search.
    This uses the fact that metrics can be computed efficiently as a function of TP,FP,FN and TN.
    By sorting the data per assigned group  we can efficiently compute these four values by looking
    at the cumlative sum of positive and negative labelled data (provided by ordered encode).
    This brings any subsequent computation of metrics down to O(1) in the dataset size.
    Parameters
    ----------
    accum_count:
    bottom: a single number or per group numpy array indicating where the grid should start
    top: a single number or per group numpy array indicating where the grid should stop
    metric1: a BaseGroupMetric
    metric2: a BaseGroupMetric
    steps: (optional) The number of divisions per group
    returns
    -------
    a tupple of three numpy arrays:
        1. the scores of metric1 and metric 2 computed for each choice of threshold
        2. the indicies corresponding to thresholds
        3. the step offset used.
    """
    groups = len(bottom)

    step = [(t - b) / steps for t, b in zip(top, bottom)]

    mesh_indices = [np.unique(np.floor(np.arange(np.floor(b), np.ceil(t + 1),
                                                 max(1, s))).astype(int))
                    for b, t, s in zip(bottom, top, step)]
    mesh = np.meshgrid(*mesh_indices, sparse=True)

    acc1=build_grid_inner(accum_count1, mesh, groups)
    acc2=build_grid_inner(accum_count2, mesh, groups)
    met1 = metric1(acc1)
    assert len(met1.shape) == 1
    met2 = metric2(acc2)
    assert len(met2.shape) == 1
    score = np.stack((met1, met2), 0)
    return score, mesh_indices, np.maximum(1, np.asarray(step))


def condense(thresholds: np.ndarray, labels: np.ndarray, lmax: int, groups: np.ndarray,
             gmax: int) -> Tuple[np.ndarray, np.ndarray]:
    """Take an array of float thresholds and non-negative integer labels, groups and
    return a sorted List of unique thresholds and the counts for each unique count of
    threshold, label, group
    parameters
    ----------
    thresholds: a numpy array of initial thresholds to reduce.
    labels: a numpy array of initial labels to count when reducing.
    lmax: the maximum label ever used.
    groups: a numpy array of initial groups to count when reducing.
    gmax: the maximum group ever used
    returns
    -------
    1. a sorted numpy array of unique thresholds
    2. corresponding counts
    """
    assert thresholds.shape == labels.shape == groups.shape
    groups = groups.astype(int)
    labels = labels.astype(int)
    unique_thresh, index = np.unique(thresholds, return_inverse=True)
    out = np.zeros((unique_thresh.shape[0], lmax, gmax))
    np.add.at(out, (index, labels, groups), 1)

    assert out.sum() == labels.shape[0]

    return unique_thresh[::-1], out[::-1]

def condense_weights(thresholds: np.ndarray, labels: np.ndarray, lmax: int, groups: np.ndarray,
             gmax: int,*, weight1=1, weight2=1) -> Tuple[np.ndarray, np.ndarray]:
    """Take an array of float thresholds and non-negative integer labels, groups and
    return a sorted List of unique thresholds and the counts for each unique count of
    threshold, label, group
    parameters
    ----------
    thresholds: a numpy array of initial thresholds to reduce.
    labels: a numpy array of initial labels to count when reducing.
    lmax: the maximum label ever used.
    groups: a numpy array of initial groups to count when reducing.
    gmax: the maximum group ever used
    weights1 and weights2
    returns
    -------
    1. a sorted numpy array of unique thresholds
    2. two corresponding counts
    """
    assert thresholds.shape == labels.shape == groups.shape
    groups = groups.astype(int)
    labels = labels.astype(int)
    unique_thresh, index = np.unique(thresholds, return_inverse=True)
    out1 = np.zeros((unique_thresh.shape[0], lmax, gmax))
    out2 = np.zeros_like(out1)
    
    np.add.at(out1, (index, labels, groups), weight1)
    np.add.at(out2, (index, labels, groups), weight2)
    
    #assert out1.sum() == labels.shape[0]
    #assert out2.sum() == labels.shape[0]
    return unique_thresh[::-1], out1[::-1], out2[::-1]

def test_cum_sum(accum_count, groups):
    "Check expected properties of accum_count hold"
    # N.B. all values are int, and float approximation is not a concern
    for group in range(groups):
        assert (np.abs(accum_count[group].sum(1) - accum_count[group][0].sum(0)).sum()) == 0
        # Total sum must be the same
        assert (np.abs(accum_count[group][:, 0] + accum_count[group][:, 2]
                       - accum_count[group][0][0] - accum_count[group][0][2]).sum()) == 0
        # TP+FN must be the same
        assert (np.abs(accum_count[group][:, 1] + accum_count[group][:, 3]
                       - accum_count[group][0][1] - accum_count[group][0][3]).sum()) == 0
        # FP+TN must be the same


def cumsum_zero(array: np.ndarray):
    "compute a cumalitive sum starting with zero (i.e. the sum upto the first element)"
    zero = np.zeros((1,) + array.shape[1:], dtype=int)
    out = np.concatenate((zero, np.cumsum(array, 0)), 0)
    return out


def grid_search_no_weights(ordered_encode, ass_size, score,
                           metric1, metric2, steps, directions):
    """Internal helper for grid search. 
    The weighted pathway requires x2 memory and computation so instead of compressing the cases 
    and computing unweighted as weighted with weights 1, we preserve the old pathway."""
   
    accum_count = [np.concatenate((cumsum_zero(o), cumsum_zero(o[::-1])[::-1]), 1)
                   for o in ordered_encode]
    # The above is the important code
    # accum_count is a list of size groups where each element is an array consisting of the number
    # of true positives, false positives, false negatives and false positives if a threshold is set
    # at a particular value. It is of size (4, groups) because the group assignment may come at test
    # time from an inaccurate classifier

    test_cum_sum(accum_count, ass_size)
    # now for the computational bottleneck
    bottom = np.zeros(ass_size)
    top = np.asarray([s.shape[0] for s in ordered_encode])
    score, mesh_indices, step = build_grid(accum_count, bottom, top, metric1, metric2, steps=steps)

    indicies = np.asarray(np.meshgrid(*mesh_indices, sparse=False)).reshape(ass_size, -1)

    front, index = keep_front(score, indicies, directions)
    if index.shape[1] > 4:
        tindex = index[:, 1:-1]
    else:
        tindex = index
    bottom = np.floor(np.maximum(step / 2, tindex.min(1) - step))
    top = np.ceil(np.minimum(top, tindex.max(1) + step))
    score, mesh_indices, _ = build_grid(accum_count, bottom, top, metric1, metric2, steps=steps)

    indicies = np.asarray(np.meshgrid(*mesh_indices, sparse=False)).reshape(ass_size, -1)
    return score, indicies, front, index

def grid_search_weights(ordered_encode, ordered_encode2, groups, score,
                        metric1, metric2, steps, directions):
    """Internal helper for grid search. 
    The weighted pathway requires x2 memory and computation so instead of compressing the cases 
    and computing unweighted as weighted with weights 1, we preserve the old pathway.
    This is the new pathway for weights"""

    accum_count1 = [np.concatenate((cumsum_zero(o), cumsum_zero(o[::-1])[::-1]), 1)
                   for o in ordered_encode]

    accum_count2 = [np.concatenate((cumsum_zero(o), cumsum_zero(o[::-1])[::-1]), 1)
                   for o in ordered_encode2]
    # The above is the important code
    # accum_count is a list of size groups where each element is an array consisting of the number
    # of true positives, false positives, false negatives and false positives if a threshold is set
    # at a particular value. It is of size (4, groups) because the group assignment may come at test
    # time from an inaccurate classifier

    #test_cum_sum(accum_count, groups)
    # now for the computational bottleneck
    bottom = np.zeros(groups)
    top = np.asarray([s.shape[0] for s in ordered_encode])
    score, mesh_indices, step = build_grid2(accum_count1, accum_count2, bottom, top, metric1,
                                            metric2, steps=steps)

    indicies = np.asarray(np.meshgrid(*mesh_indices, sparse=False)).reshape(groups, -1)

    front, index = keep_front(score, indicies, directions)
    if index.shape[1] > 4:
        tindex = index[:, 1:-1]
    else:
        tindex = index
    bottom = np.floor(np.maximum(step / 2, tindex.min(1) - step))
    top = np.ceil(np.minimum(top, tindex.max(1) + step))
    score, mesh_indices, _ = build_grid2(accum_count1, accum_count2, bottom, top, metric1,
                                         metric2, steps=steps)

    indicies = np.asarray(np.meshgrid(*mesh_indices, sparse=False)).reshape(groups, -1)
    return score, indicies, front, index


def grid_search(y_true: np.ndarray, proba: np.ndarray, metric1: Callable, metric2: Callable,
                hard_assignment: np.ndarray, true_groups: np.ndarray, *, directions=(+1, +1),
                group_response=False, steps=25,factor=None) -> Tuple[np.ndarray, np.ndarray]:
    """Efficient grid search.
    Functions under the assumtion data is hard assigned by a group classifer with errors
    and the alignment need not perfectly correspond to groups
    parameters
    ----------
    y_true: a numpy array containing the target labels
    proba: a numpy array containing the soft classifier responses.
    metric1: a BaseGroupMetric
    metric2: a BaseGroupMetric
    hard_assignment: a potentially lossy assignment of datapoints to groups by a classifier.
    true_groups: a numpy array containing the actual group assignment
    group_response: (optional) The response used by a classifier to soft assign groups.
    steps: (optional) The number of divisions per group
    directions: (optional) a binary vector containing [+1,-1] indicating if greater or lower
        solutions are prefered
    """
    assert proba.shape[1] == 2
    assert proba.ndim == 2
    assert y_true.ndim == 1
    assert y_true.shape[0] == proba.shape[0]
    points = y_true.shape[0]
    assert hard_assignment.shape[0] == points
    assert hard_assignment.ndim == 1
    assert true_groups.shape[0] == points
    assert true_groups.ndim == 1
    score = proba[:, 0] - proba[:, 1]
    if group_response is not False:
        assert group_response.ndim == 1
        assert points == group_response.shape[0]
        score /= group_response  # generally not useful

    unweighted_path = metric1.cond_weights is None and metric2.cond_weights is None

    # hard assignment and true groups need to be ints
    # All supported metrics are invariant to the peturbation of true groups
    # ordering. So we do not need to assume that they arrive as ints, we just encode
    # and discard the ordering later.

    encoder = OrdinalEncoder()
    encoder.fit(true_groups.reshape(-1, 1))
    true_groups = encoder.transform(true_groups.reshape(-1, 1)).reshape(-1).astype(int)

    if true_groups.max() > hard_assignment.max():
        logger.warning('Fewer groups used in infered groups than in the true groups')
    elif true_groups.max() < hard_assignment.max():
        logger.warning('Fewer groups used in true groups, than in the infered groups')
    assigned_labels = np.unique(hard_assignment) 
    ass_size = assigned_labels.shape[0]
    groups = true_groups.max() + 1
    assigned_groups = hard_assignment.max() + 1
    # Preamble that reorganizes the data for efficient computation
    # This uses lists indexed by group rather than arrays
    # as there are different amounts of data per group

    masks = [hard_assignment == g for g in assigned_labels]
    if unweighted_path:
        collate = [condense(score[m], y_true[m], 2, true_groups[m], groups) for m in masks]
    else:
        assert factor is not None, 'Called fit with conditional metrics but no conditional factor provided'
        #Consider disabling this and just use weight=1 if no factor provided
        weight1 = 1
        weight2 = 1
        if metric1.cond_weights is not None:
            weight1=metric1.cond_weights(factor, true_groups, y_true)
        if metric2.cond_weights is not None:
            weight2=metric2.cond_weights(factor, true_groups, y_true)
        def mask_weight(weight,mask):
            if isinstance(weight,np.ndarray):
                return weight[mask]
            else:
                return weight

        collate = [condense_weights(score[m], y_true[m], 2, true_groups[m],
                                    groups, weight1=mask_weight(weight1,m), 
                                            weight2=mask_weight(weight2,m)) for m in masks]
    thresholds = [c[0] for c in collate]
    ordered_encode = [c[1] for c in collate]
    if unweighted_path:
        ordered_encode2 = False
    else:
        ordered_encode2 = [c[1] for c in collate]

    thresholds = [np.concatenate((t[0:1] + 1e-4, t), 0) for t in thresholds]
    # add threshold above maximum value

    if unweighted_path:
        score, indicies, front, index = grid_search_no_weights(ordered_encode, ass_size, score,
                           metric1, metric2, steps, directions)
    else:
        score, indicies, front, index = grid_search_weights(ordered_encode, ordered_encode2,
                                                            ass_size, score, metric1, metric2,
                                                            steps, directions)


    new_front, new_index = keep_front(score, indicies, directions)
    # merge the two existing fronts
    front, index = keep_front(np.concatenate((front, new_front), 1),
                              np.concatenate((index, new_index), 1),
                              directions)
    selected_thresholds = np.asarray([(g[i] + g[np.minimum(g.shape[0] - 1, i + 1)]) / 2
                                      for g, i in zip(thresholds, index)])
    return front, selected_thresholds