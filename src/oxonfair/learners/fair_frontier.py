"""Slow pathway for computing fairness constraints. Compatable with Scorers and group metrics,
while efficient_compute is only compatable with group metrics"""
from typing import Callable, Tuple
import numpy as np
from .efficient_compute import keep_front
from ..utils.scipy_metrics_cont_wrapper import ScorerRequiresContPred
AUTOGLUON_EXISTS = True
try:
    from autogluon.core.metrics import Scorer
except ModuleNotFoundError:
    AUTOGLUON_EXISTS = False


def compute_metric(metric: Callable, y_true: np.ndarray, proba: np.ndarray,
                   threshold_assignment: np.ndarray,
                   weights: np.ndarray) -> np.ndarray:
    """takes probability scores, and offsets them according to the weights * threshold_assignment.
        then select the max and compute a fairness metric """
    score = np.zeros((weights.shape[-1]))
    y_true = np.asarray(y_true)
    threshold_assignment = np.asarray(threshold_assignment)

    pass_scores = (isinstance(metric, ScorerRequiresContPred) or
                   (AUTOGLUON_EXISTS and isinstance(metric, Scorer) and (metric.needs_pred is False)))
    # Preallocate because this next loop is the system bottleneck
    tmp = np.empty((threshold_assignment.shape[0], weights.shape[1]), dtype=threshold_assignment.dtype)
    pred = np.empty(threshold_assignment.shape[0], dtype=int)
    for i in range(weights.shape[-1]):
        np.dot(threshold_assignment, weights[:, :, i], tmp)
        if pass_scores is False:
            tmp += proba
            np.argmax(tmp, -1, pred)
            score[i] = metric(y_true, pred)[0]
        else:
            tmp += proba
            tmp[:, 1] -= tmp[:, 0]
            score[i] = metric(y_true, tmp[:, 1])
    return score


def sort_by_front(front: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    "sort the front and weights according to front[0]"
    sort_ind = np.argsort(front[0])
    weights = weights[:, :, sort_ind]
    front = front[:, sort_ind]
    return front, weights

# Solution modified from here:
# https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python


def keep_front_old(solutions: np.ndarray, initial_weights: np.ndarray, directions: np.ndarray,
               additional_constraints,
               *, tol=1e-12) -> Tuple[np.ndarray, np.ndarray]:
    """Modified from
        https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
        Returns Pareto efficient row subset of solutions and its associated weights
        Direction is a vector that governs if the frontier should maximize or minimize each
        direction.
        Where an element of direction is positive frontier maximizes, negative, it mimizes.
        """
    front = solutions.T.copy()
    front *= directions

    weights = initial_weights.T.copy()
    # check if additional_constraints exist, and if so use them to filter the data.
    # if additional_constraints is not None:
        
    # sort points by decreasing sum of coordinates
    # Add 10**-6 * magnitude of weights so that in the event of a near tie, pick points close
    # to 0 first
    order = (front.sum(1) - (10**-6) * np.abs(weights).sum(1).sum(1)).argsort()[::-1]
    front = front[order]
    weights = weights[order]
    # initialize a boolean mask for currently undominated points
    undominated = np.ones(front.shape[0], dtype=bool)
    for i in range(front.shape[0]):
        # process each point in turn
        points_left = front.shape[0]
        if i >= points_left:
            break
        # find all points not dominated by i
        # since points are sorted by coordinate sum
        # i cannot dominate any points in 1,...,i-1
        undominated[i] = True  # Bug fix missing from online version
        undominated[i + 1:points_left] = (front[i + 1:] >= front[i] + tol).any(1)
        # keep points undominated so far
        front = front[undominated[:points_left]]
        weights = weights[undominated[:points_left]]
    weights = weights.T
    if directions is not False:
        front *= directions
    front = front.T
    front, weights = sort_by_front(front, weights)

    return front, weights


def linear_interpolate(front: np.ndarray, weights: np.ndarray, gap=0.01) -> np.ndarray:
    """we want the points found to cover the frontier i.e. there should be no big gaps in w.
        we achieve this by linearly interpolating in the frontier, and using this to determine
        step size in w """
    eps = (front.max(1) - front.min(1)) * gap  # step size in the frontier
    out = [np.linspace(weights[:, :, i], weights[:, :, i + 1],
                       num=int((np.abs(front[:, i + 1] - front[:, i]) / eps).max()) + 1)
           for i in range(weights.shape[-1] - 1)]
    out = np.concatenate(out, 0).transpose(1, 2, 0)
    return out


def sigmoid(x):
    "broadcastable sigmoid function"
    return 1.0 / (1.0 + np.exp(-x))


def inv_sigmoid(x):
    "broadcastable inverse sigmoid function"
    assert (x < 1).all()
    assert (x > 0).all()
    return np.log(x / (1 - x))


def make_grid_between_points(point_a: np.ndarray, point_b: np.ndarray, *, refinement_factor=2,
                             add_zero=False, use_linspace=True,
                             logit_scaling=False) -> np.ndarray:
    """
    creates new grid points between two points by refining each axis in which the two points do not
    coincide; the refinement_factor defines into how many parts such an axis is divided
    """
    assert point_a.shape == point_b.shape
    groups, classes = point_a.shape
    point_a = point_a.flatten()
    point_b = point_b .flatten()
    mins = np.minimum(point_a, point_b)
    maxs = np.maximum(point_a, point_b)

    if use_linspace:
        if logit_scaling:
            maxs = sigmoid(maxs)
            mins = sigmoid(mins)
        diffs = maxs - mins
        if logit_scaling:
            epsilon = 0.05
        else:
            if any(diffs > 0):
                epsilon = diffs[diffs > 0].min()
            else:
                epsilon = 0.05
        mins -= epsilon
        maxs += epsilon
        if logit_scaling:
            maxs = np.minimum(maxs, 1 - epsilon)
            mins = np.maximum(mins, epsilon)
    else:
        epsilon = refinement_factor.flatten()
        mins -= epsilon * 1.5
        maxs += epsilon * 1.51

    zero = np.zeros((1))
    if logit_scaling:
        zero = 0.5 * np.ones(1)
    if use_linspace:
        if logit_scaling:
            axx = [inv_sigmoid(np.linspace(mins[i], maxs[i], num=refinement_factor + 1))
                   for i in range(maxs.shape[0])]
        else:
            axx = [(np.linspace(mins[i], maxs[i], num=refinement_factor + 1))
                   for i in range(maxs.shape[0])]
    else:
        axx = [np.arange(mins[i], maxs[i], step=epsilon[i]) for i in range(maxs.shape[0])]

    if add_zero:
        axx = [np.concatenate((ax, zero), 0) for ax in axx]

    mesh = np.meshgrid(*axx, copy=False)
    shape = (groups, classes + 1) + mesh[0].shape
    weights = np.zeros(shape, dtype=np.float16)
    for i in range(classes):  # Ignore final class -- the space of thresholds is overparameterized
        for j in range(groups):
            weights[j, i] = mesh[(classes) * j + i]
    weights = weights.reshape((weights.shape[0], weights.shape[1], -1))
    assert not np.isnan(weights).any()
    return weights


def make_finer_grid(weights: np.ndarray, refinement_factor=2, use_linspace=True) -> np.ndarray:
    """
    creates additional grid points between two consecutive points in the current weights set; see
    the function make_grid_between_points below for the meaning of the refinement_factor
    """
    new_weights = [make_grid_between_points(weights[:, :-1, ell],
                                            weights[:, :-1, ell + 1],
                                            refinement_factor=refinement_factor,
                                            use_linspace=use_linspace)
                   for ell in range(weights.shape[-1] - 1)]
    output = np.concatenate(new_weights, axis=2)
    output = np.concatenate((output, np.zeros((output.shape[0], output.shape[1], 1), dtype=np.float16)), axis=2)
    output = np.unique(output, axis=-1)

    return output


def front_from_weights(weights: np.ndarray, y_true: np.ndarray, proba: np.ndarray,
                       groups_infered: np.ndarray,
                       tupple_metrics) -> np.ndarray:
    """Computes the values of each metric from the weights"""
    front = np.stack(list(map(lambda x: compute_metric(x, y_true, proba,
                                                       groups_infered, weights), tupple_metrics)))
    return front


def build_coarse_to_fine_front(metrics: Tuple[callable],
                               y_true: np.ndarray,
                               proba: np.ndarray,
                               groups_infered: np.ndarray,
                               *,
                               directions=(+1, +1),
                               initial_divisions=15,
                               nr_of_recursive_calls=5,
                               refinement_factor=4,
                               logit_scaling=False,
                               existing_weights=None,
                               additional_constraints=()) -> Tuple[np.ndarray, np.ndarray]:
    """
    this function performs coarse-to-fine grid-search for computing the Pareto front
    """
    assert len(metrics) >= 2
    assert groups_infered.ndim == 2
    assert nr_of_recursive_calls > 0
    groups = groups_infered.shape[1]

    classes = proba.shape[1] - 1  # n.b. this is really classes-1
    upper_bound = (proba[:, :classes] - proba.min(1)[:, np.newaxis]).max(0)
    lower_bound = (proba[:, :classes] - proba.max(1)[:, np.newaxis]).min(0)
    min_initial = np.ones((groups, classes))
    min_initial[:, :] = lower_bound[:, np.newaxis]
    max_initial = np.ones((groups, classes))
    max_initial[:, :] = upper_bound[:, np.newaxis]
    # perform an initial two stage search, first coarsely over every possible value
    # then take the front and search over valid values from it
    weights = make_grid_between_points(min_initial, max_initial,
                                       refinement_factor=initial_divisions - 1,
                                       logit_scaling=logit_scaling)
    front = front_from_weights(weights, y_true, proba, groups_infered, metrics)
    front, weights = keep_front(front, weights, directions, additional_constraints)
    # second stage
    mins = weights[:, :-1].min(-1)  # drop zeros
    maxs = weights[:, :-1].max(-1)
    mins -= 2 / initial_divisions  # if we only get one point, expand around it
    maxs += 2 / initial_divisions
    eps = ((maxs - mins))
    new_weights = make_grid_between_points(mins, maxs, refinement_factor=initial_divisions,
                                           add_zero=True, logit_scaling=logit_scaling)
    new_front = front_from_weights(new_weights, y_true, proba, groups_infered, metrics)
    weights = np.concatenate((new_weights, weights), -1)
    front = np.concatenate((new_front, front), -1)
    if existing_weights is not None:
        existing_weights = existing_weights.astype(weights.dtype)
        existing_front = front_from_weights(existing_weights, y_true, proba, groups_infered, metrics)
        weights = np.concatenate((existing_weights, weights), -1)
        front = np.concatenate((existing_front, front), -1)

    front, weights = keep_front(front, weights, directions, additional_constraints)
    for _ in range(nr_of_recursive_calls - 1):
        if weights.shape[-1] != 1:
            eps /= refinement_factor
            new_weights = make_finer_grid(weights, eps, use_linspace=False)
            new_front = front_from_weights(new_weights, y_true, proba, groups_infered,
                                           metrics)
            weights = np.concatenate((new_weights, weights), -1)
            front = np.concatenate((new_front, front), -1)
            front, weights = keep_front(front, weights, directions, additional_constraints)

    # densify the front with uniform interpolation
    if weights.shape[-1] > 1:
        weights = linear_interpolate(front, weights, gap=0.02)
        front = front_from_weights(weights, y_true, proba, groups_infered, metrics)
        front, weights = keep_front(front, weights, directions, additional_constraints)

    return front, weights
