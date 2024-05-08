"""The clases used to define group measures for fairness and performance"""

import abc
from abc import abstractmethod
import logging
import copy

from typing import Callable, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class BaseGroupMetric:
    """Building block for GroupMetrics.
    It performs book keeping allowing group metrics to take as raw
    input either a single array containing t_pos,f_pos,f_neg,t_neg values broadcast over groups and
    many different thresholds, or singular vectors corresponding to y_true, y_pred, and groups.
    Also contains additional annotations: name, and greater_is_better
    """

    __metaclass__ = abc.ABCMeta

    def __init__(
        self,
        func: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        name: str,
        greater_is_better: bool,
        cond_weights=None,
    ) -> None:
        self.func: Callable[
            [np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray
        ] = func
        self.name: str = name
        self.greater_is_better: bool = greater_is_better
        if cond_weights is None:
            self.cond_weights = None
        else:
            assert isinstance(
                cond_weights, ConditionalWeighting
            ), "cond_weights must be a Conditional Metric"
            self.cond_weights = cond_weights

    @abstractmethod
    def __call__(self, *args: np.ndarray) -> np.ndarray:
        pass

    def build_array(
        self, args: Tuple[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Helper Function for all child classes.
        Allows the overloading of GroupMetrics so they can be used both in the inner loop of
        efficient_compute.py and to return scores on raw data.
        parameters
        ----------
        args: a Tuple of numpy arrays.
            Either a Tuple containing a single (4 x entries x groups) or
                a Tuple containing 3 vectors of the same length corresponding to y_true, y_pred,
                and groups
        returns
            4  (entries x groups) sized arrays, where entries is 1 if args consisted of 3 vectors.
        """
        if len(args) == 1:
            if args[0].shape[0] != 4:
                logger.error(
                    "Only one argument passed to group metric, but the first dimension is not 4."
                )
            return args[0][3], args[0][2], args[0][1], args[0][0]
            # N.B. order reversed
        if len(args) != 3 and len(args) != 4:
            logger.error(
                "Group metrics can take either one, three, or four broadcast arrays"
            )

        if len(args) == 4 and self.cond_weights is None:
            logger.error(
                """Metric called with four inputs, indicating that we should condition,
                but no conditioning function provided."""
            )

        y_true: np.ndarray = args[0].astype(int)
        y_pred: np.ndarray = args[1].astype(int)
        groups: np.ndarray = args[2]
        if len(args) == 4:
            weights = self.cond_weights(args[3], groups, y_true)
        else:
            weights = False

        assert y_true.size == y_pred.size == groups.size, "Inputs to group_metric are of different sizes. Make sure that all variables are ordinal encoded and not one-hot."
        t_pos = y_true * y_pred
        f_pos = (1 - y_true) * y_pred
        f_neg = y_true * (1 - y_pred)
        t_neg = (1 - y_true) * (1 - y_pred)
        unique = np.unique(groups)
        out = np.zeros((4, 1, unique.shape[0]))
        for i, group_name in enumerate(unique):
            mask = groups == group_name
            if weights is False:
                out[0, :, i] = t_pos[mask].sum()
                out[1, :, i] = f_pos[mask].sum()
                out[2, :, i] = f_neg[mask].sum()
                out[3, :, i] = t_neg[mask].sum()
            else:
                w = weights[mask]
                out[0, :, i] = t_pos[mask].dot(w)
                out[1, :, i] = f_pos[mask].dot(w)
                out[2, :, i] = f_neg[mask].dot(w)
                out[3, :, i] = t_neg[mask].dot(w)
        return out[0], out[1], out[2], out[3]

    def clone(self, new_name: str, cond_weights=False):
        """Generates a copy of self with a new name, and (optionally) a new cond_weights
        Use cond_weights = False to use the existing cond_weights in the metric being cloned.
            cond_weights =  None to remove it
            cond_weights = ConditionalMetric in any other case"""
        out = copy.copy(self)
        out.name = new_name
        if cond_weights is not False:
            out.cond_weights = cond_weights
        return out


class PerGroup(BaseGroupMetric):
    "Helper class for reporting scores per group"

    def __call__(self, *args: np.ndarray) -> np.ndarray:
        array = self.build_array(args)
        val = self.func(*array)
        return val


class GroupMax(BaseGroupMetric):
    "Helper class for reporting maximal score of any  group"

    def __call__(self, *args: np.ndarray) -> np.ndarray:
        array = self.build_array(args)
        val = self.func(*array)
        return val.max(-1)


class GroupMin(BaseGroupMetric):
    "Helper class for reporting minimal score of any group"

    def __call__(self, *args: np.ndarray) -> np.ndarray:
        array = self.build_array(args)
        val = self.func(*array)
        return val.min(-1)


class GroupMaxDiff(BaseGroupMetric):
    "Helper class for reporting maximal difference in score between any pair of groups"

    def __call__(self, *args: np.ndarray) -> np.ndarray:
        array = self.build_array(args)
        val = self.func(*array)
        return val.max(-1) - val.min(-1)


class GroupDiff(BaseGroupMetric):
    "Helper class for reporting average difference in score between all pairs of groups"

    def __call__(self, *args: np.ndarray) -> np.ndarray:
        array = self.build_array(args)
        val = self.func(*array)
        broadcast = val[:, np.newaxis, :] - val[:, :, np.newaxis]
        trunc = np.maximum(broadcast, 0)
        collate = trunc.sum(1).sum(1) / (val.shape[1] * (val.shape[1] - 1) / 2)
        return collate


class GroupRatio(BaseGroupMetric):
    "Helper class for reporting average score ratio  between any pair of groups"

    def __call__(self, *args: np.ndarray) -> np.ndarray:
        array = self.build_array(args)
        val = self.func(*array)
        broadcast = val[:, np.newaxis, :] / np.maximum(1e-12, val[:, :, np.newaxis])
        trunc = np.minimum(broadcast, 1.0/np.maximum(1e-12, broadcast))
        collate = (trunc.sum(1).sum(1)-trunc.shape[1]) / (val.shape[1] * (val.shape[1] - 1))
        return collate

class GroupMinimalRatio(BaseGroupMetric):
    "Helper class for reporting minimal score ratio  between any pair of groups"

    def __call__(self, *args: np.ndarray) -> np.ndarray:
        array = self.build_array(args)
        val = self.func(*array)
        return val.min(-1) / np.maximum(1e-12, val.max(-1))


class Overall(BaseGroupMetric):
    "Helper class for reporting score over entire dataset"

    def __call__(self, *args: np.ndarray) -> np.ndarray:
        t_pos, f_pos, f_neg, t_neg = self.build_array(args)
        val = self.func(t_pos.sum(1), f_pos.sum(1), f_neg.sum(1), t_neg.sum(1))
        return val


class GroupAverage(BaseGroupMetric):
    "Helper class for reporting scores averaged over groups"

    def __call__(self, *args: np.ndarray) -> np.ndarray:
        array = self.build_array(args)
        val = self.func(*array)
        return val.mean(-1)


class GroupMetric(BaseGroupMetric):
    """Broadcastable metrics used by efficient compute.
    All methods either takes a single 3d numpy array as input or three vectors:
    y_true, y_pred, and groups
    The matrix passed to any function is assumed to be of size
    4 x entries x groups.
    The first entry of the first axis corresponds to the number of True Negatives,
    second False Negatives,
    third False Positives, and
    fourth True Positives.

    init parameters:
    func: a function that takes 4 numpy arrays corresponding to:
        True Positives, False Positives, False Negatives, and True Negatives as an input,
        and returns a numpy array of scores.
    name: a string description of the score.
    greater_is_better: a bool indicating if the score should be maximised or minimised.
    """

    def __init__(
        self,
        func: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        name: str,
        greater_is_better: bool = True,
        cond_weights=None,
    ) -> None:
        super().__init__(func, name, greater_is_better, cond_weights=cond_weights)
        self.max: GroupMax = GroupMax(
            func,
            "Maximal Group " + name,
            greater_is_better=greater_is_better,
            cond_weights=cond_weights,
        )
        self.min: GroupMin = GroupMin(
            func,
            "Minimal Group " + name,
            greater_is_better=greater_is_better,
            cond_weights=cond_weights,
        )
        self.overall: Overall = Overall(
            func,
            "Overall " + name,
            greater_is_better=greater_is_better,
            cond_weights=cond_weights,
        )
        self.average: GroupAverage = GroupAverage(
            func,
            "Average Group " + name,
            greater_is_better=greater_is_better,
            cond_weights=cond_weights,
        )
        self.diff: GroupDiff = GroupDiff(
            func,
            "Average Group Difference in " + name,
            greater_is_better=False,
            cond_weights=cond_weights,
        )
        self.max_diff: GroupMaxDiff = GroupMaxDiff(
            func,
            "Maximal Group Difference in " + name,
            greater_is_better=False,
            cond_weights=cond_weights,
        )
        self.ratio: GroupRatio = GroupRatio(
            func,
            "Average Group Ratio in " + name,
            greater_is_better=True,
            cond_weights=cond_weights,
        )
        self.min_ratio: GroupMinimalRatio = GroupMinimalRatio(
            func,
            "Minimal Group Ratio in " + name,
            greater_is_better=True,
            cond_weights=cond_weights,
        )
        self.per_group: PerGroup = PerGroup(
            func,
            "Per Group " + name,
            greater_is_better=greater_is_better,
            cond_weights=cond_weights,
        )

    def clone(self, new_name, cond_weights=False):
        my_type = self.__class__
        if cond_weights is False:
            out = my_type(
                self.func, new_name, self.greater_is_better, self.cond_weights
            )
        else:
            out = my_type(self.func, new_name, self.greater_is_better, cond_weights)
        return out

    def __call__(self, *args: np.ndarray) -> np.ndarray:
        return self.overall(*args)


class AddGroupMetrics(BaseGroupMetric):
    """Group Metric consisting of the weighted sum of two existing metrics
    parameters
    ----------
    metric1: a BaseGroupMetric
    metric2: a BaseGroupMetric
    name:    a string
    weight: (optional) a float between 0 and 1.
    returns
    -------
    a BaseGroupMetric that gives scores of the form:
        weight*metric1_response+(1-weight)*metric2_response"""

    def __init__(
        self,
        metric1: BaseGroupMetric,
        metric2: BaseGroupMetric,
        name: str,  # pylint: disable=super-init-not-called
        weight: float = 0.5,
    ) -> None:
        self.metric1: BaseGroupMetric = metric1
        self.metric2: BaseGroupMetric = metric2
        self.name = name
        self.cond_weights = None
        if metric1.greater_is_better != metric2.greater_is_better:
            logger.error(
                "Metric1 and metric2  must satisfy the condition. metric1.greater_is_better == metric2.greater_is_better "
            )
        if not 0 <= weight <= 1:
            logger.error("Weight must be between 0 and 1")
        self.weight: float = weight
        self.greater_is_better = metric1.greater_is_better

    def __call__(self, *args: np.ndarray) -> np.ndarray:
        return self.weight * self.metric1(*args) + (1 - self.weight) * self.metric2(
            *args
        )


class Utility(GroupMetric):
    """A group metric for encoding utility functions.
    See Fairness on the Ground: htt_poss://arxiv.org/pdf/2103.06172.pdf
    This is implemented as a group metric, so the standard fairness concerns i.e.
    difference in utility between groups, ratio of utility, minimum utility of any group
    are all supported.
    Parameters
    ----------
    utility: a sequence of length 4 corresponding to the cost of true positives,
             false positive, false negatives, and true negatives
    name: a string corresponding to the name of the utility function
    greater_is_better: a bool indicating if the utility should be maximised or minimized
    """

    def __init__(self, utility, name, greater_is_better=False):
        if len(utility) != 4:
            logger.error("Utility vector must be of length 4.")
        self.utility = utility
        super().__init__(self.cost, name, greater_is_better)

    def cost(self, t_pos, f_pos, f_neg, t_neg):
        "Method for computing the cost/utility"
        return (
            t_pos * self.utility[0]
            + f_pos * self.utility[1]
            + f_neg * self.utility[2]
            + t_neg * self.utility[3]
        ) / (t_pos + f_pos + f_neg + t_neg)


class ConditionalWeighting:
    """This is used to implement a range of conditional metrics analgous to those set out in Why fairness can not be automated and statistics
    (cite properly)
    The metrics are useful when you want to (for example) to ensure that individual schools should not discriminate in aggrigate
    against men and women, but different schools may have different acceptance rates, and have different rates of men
    and women applying there.
    """

    def __init__(self, per_group):
        self.per_group = per_group

    def __call__(
        self, conditioning_factor: np.array, groups: np.array, y_true: np.array
    ) -> np.array:
        assert conditioning_factor.shape == y_true.shape
        assert groups.shape == y_true.shape
        weights = np.zeros_like(y_true, dtype=float)
        uniq_f = np.unique(conditioning_factor)
        uniq_g = np.unique(groups)

        factor_masks = {}
        factor_weights = {}
        total_pos = y_true.sum()
        total_neg = y_true.shape[0] - total_pos
        for f in uniq_f:
            mask = conditioning_factor == f
            factor_masks[f] = mask
            factor_pos = y_true[mask].sum()
            factor_neg = mask.sum() - factor_pos
            factor_weights[f] = self.per_group(
                total_pos, total_neg, factor_pos, factor_neg
            )
        for g in uniq_g:
            mask = groups == g
            group_pos = y_true[mask].sum()
            group_neg = mask.sum() - group_pos
            for f in uniq_f:
                new_mask = mask * factor_masks[f]
                factor_sub_pos = y_true[new_mask].sum()
                factor_sub_neg = new_mask.sum() - factor_sub_pos
                weights[new_mask] = (
                    self.per_group(group_pos, group_neg, factor_sub_pos, factor_sub_neg)
                    / factor_weights[f]
                )
        return weights
