"""Slow pathway for computing fairness constraints. Compatable with Scorers and group metrics,
while efficient_compute is only compatable with group metrics"""
from typing import Callable, Tuple, Sequence, Optional, Union
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


def compute_metrics(metrics: Sequence[Callable], y_true: np.ndarray, proba: np.ndarray,
                    threshold_assignment: np.ndarray,
                    weights: np.ndarray) -> np.ndarray:
    """takes probability scores, and offsets them according to the weights * threshold_assignment.
        then select the max and compute a fairness metric """
    scores = np.zeros((len(metrics), weights.shape[-1]))
    y_true = np.asarray(y_true)
    # assert weights[:, 1, :].sum() == 0
    weights = weights[:, 0, :]
    proba = proba[:, 0] - proba[:, 1]

    threshold_assignment = np.asarray(threshold_assignment)

    pass_scores = [(isinstance(metric, ScorerRequiresContPred) or
                   (AUTOGLUON_EXISTS and isinstance(metric, Scorer) and (metric.needs_pred is False)))
                   for metric in metrics]
    # Preallocate because this next loop is the system bottleneck
    tmp = np.empty((threshold_assignment.shape[0],), dtype=threshold_assignment.dtype)
    # diff = np.empty(threshold_assignment.shape[0], dtype=threshold_assignment.dtype)
    # pred = np.empty(threshold_assignment.shape[0], dtype=int)
    for i in range(weights.shape[-1]):
        threshold_assignment.dot(weights[:, i], out=tmp)
        tmp += proba  # [:, np.newaxis]
        pred = (tmp < 0)  # <= may be causing a mismatch

        # np.dot(threshold_assignment, weights[:, i], tmp)
        # tmp += proba
        for j, metric in enumerate(metrics):
            if pass_scores[j] is False:
                scores[j, i] = metric(y_true, pred)[0]
            else:
                # np.subtract(tmp[:, 1], tmp[:, 0], diff)
                scores[j, i] = metric(y_true, tmp)
    return scores


def sort_by_front(front: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    "sort the front and weights according to front[0]"
    sort_ind = np.argsort(front[0])
    weights = weights[:, :, sort_ind]
    front = front[:, sort_ind]
    return front, weights

# Solution modified from here:
# https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python


def linear_interpolate(front: np.ndarray, weights: np.ndarray, gap: float = 0.01) -> np.ndarray:
    """we want the points found to cover the frontier i.e. there should be no big gaps in w.
        we achieve this by linearly interpolating in the frontier, and using this to determine
        step size in w """
    eps = (front.max(1) - front.min(1)) * gap  # step size in the frontier
    out = [np.linspace(weights[:, :, i], weights[:, :, i + 1],
                       num=int((np.abs(front[:, i + 1] - front[:, i]) / eps).max()) + 1)
           for i in range(weights.shape[-1] - 1)]
    out_np = np.concatenate(out, 0).transpose(1, 2, 0)
    return out_np


def sigmoid(x: np.ndarray) -> np.ndarray:
    "broadcastable sigmoid function"
    return 1.0 / (1.0 + np.exp(-x))


def inv_sigmoid(x) -> np.ndarray:
    "broadcastable inverse sigmoid function"
    assert (x < 1).all()
    assert (x > 0).all()
    return np.log(x / (1 - x))


def make_grid_between_points(point_a: np.ndarray, point_b: np.ndarray, refinement_factor: Union[int, np.ndarray], *,
                             add_zero: bool = False, use_linspace: bool = True,
                             logit_scaling: bool = False) -> np.ndarray:
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
        axx = make_grid_linspace(mins, maxs, logit_scaling, refinement_factor, add_zero)
    else:
        axx = make_grid_arange(mins, maxs, refinement_factor, add_zero)

    mesh = np.meshgrid(*axx, copy=False)
    shape = (groups, classes + 1) + mesh[0].shape
    weights = np.zeros(shape, dtype=np.float16)
    for i in range(classes):  # Ignore final class -- the space of thresholds is overparameterized
        for j in range(groups):
            weights[j, i] = mesh[(classes) * j + i]
    weights = weights.reshape((weights.shape[0], weights.shape[1], -1))
    assert not np.isnan(weights).any()
    return weights


def make_grid_linspace(mins: np.ndarray, maxs: np.ndarray, logit_scaling: bool,
                       refinement_factor: Union[int, np.ndarray], add_zero: bool) -> Sequence[np.ndarray]:
    if logit_scaling:
        maxs = sigmoid(maxs)
        mins = sigmoid(mins)
    diffs = maxs - mins
    if logit_scaling:
        epsilon = 0.0
    else:
        epsilon = 0.005
        if any(diffs > 0):
            epsilon = diffs[diffs > 0].min()
    mins -= epsilon
    maxs += epsilon
    zero = np.zeros((1))
    if logit_scaling:
        zero = 0.5 * np.ones(1)
        axx = [inv_sigmoid(np.linspace(mins[i], maxs[i], num=refinement_factor + 1))
               for i in range(maxs.shape[0])]
    else:
        axx = [(np.linspace(mins[i], maxs[i], num=refinement_factor + 1))
               for i in range(maxs.shape[0])]
    if add_zero:
        axx = [np.concatenate((ax, zero), 0) for ax in axx]
    return axx


def make_grid_arange(mins: np.ndarray, maxs: np.ndarray, refinement_factor: np.ndarray,
                     add_zero: bool) -> Sequence[np.ndarray]:
    epsilon = refinement_factor.flatten()
    mins -= epsilon * 1.5
    maxs += epsilon * 1.51
    mins -= epsilon
    maxs += epsilon
    zero = np.zeros((1))
    axx = [np.arange(mins[i], maxs[i], step=epsilon[i]) for i in range(maxs.shape[0])]
    if add_zero:
        axx = [np.concatenate((ax, zero), 0) for ax in axx]
    return axx


def make_finer_grid(weights: np.ndarray, refinement_factor: Union[int, np.ndarray] = 2,
                    use_linspace: bool = True) -> np.ndarray:
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
                       tupple_metrics: Sequence[Callable]) -> np.ndarray:
    """Computes the values of each metric from the weights"""
    front = compute_metrics(tupple_metrics, y_true, proba, groups_infered, weights)
    # front = np.stack(list(map(lambda x: compute_metric(x, y_true, proba,
    #                                                   groups_infered, weights), tupple_metrics)))
    return front


def build_coarse_to_fine_front(metrics: Sequence[Callable],
                               y_true: np.ndarray,
                               proba: np.ndarray,
                               groups_infered: np.ndarray,
                               directions: np.ndarray,
                               *,
                               initial_divisions: int = 15,
                               nr_of_recursive_calls: int = 5,
                               refinement_factor: int = 4,
                               logit_scaling: bool = False,
                               existing_weights: Optional[np.ndarray] = None,
                               additional_constraints: Sequence = None) -> Tuple[np.ndarray, np.ndarray]:
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
