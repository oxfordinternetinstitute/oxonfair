"""The entry point to fair. Defines the FairPredictor object used to access fairness
functionality."""
from ast import Tuple
from typing import Optional
import copy
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from ..utils import group_metrics
from .. utils.scipy_metrics_cont_wrapper import ScorerRequiresContPred
from ..utils.group_metric_classes import BaseGroupMetric, Overall

from ..utils import performance as perf
from . import efficient_compute, fair_frontier

AUTOGLUON_EXISTS = True
try:
    from autogluon.core.metrics import Scorer
    from autogluon.tabular import TabularPredictor
except ModuleNotFoundError:
    AUTOGLUON_EXISTS = False

logger = logging.getLogger(__name__)


class FairPredictor:
    """Assess and mitigate the unfairness and effectiveness of a binary predictor
    post-fit by computing group specific metrics, and performing threshold adjustment.

    Parameters
    ----------
    predictor: a binary  predictor that will be evaluated and modified. This can be:
        1. An autogluon binary predictor.
        2. A sklearn classifier.
        3. An arbitary function
        4. The value None.
        If None  is used, we assume that we are rescoring predictions already made elsewhere, and
        the validation data should be a copy of the classifier outputs.

    validation_data: This can be:
        1. a pandas dataframe that can be read by predictor.
        2. a dict contain mutliple entries
            'data' containing a pandas dataframe or numpy array to be fed to the classifier.
            'target' the ground truth-labels used to evaluate classifier peformance.
            'groups' (optional)
            'cond_factor'
    groups (optional, default None): is an indicator of protected attributes, i.e.  the discrete
        groups used to measure fairness
    it may be:
        1. The name of a pandas column containing discrete values
        2. a vector of the same size as the validation set containing discrete values
        3. The value None   (used when we don't require groups, for example,
          if we are optimizing F1 without per-group thresholds, or if groups are explicitly
               specified by a dict in validation data)
    inferred_groups: (Optional, default False) A binary or multiclass autogluon predictor that
                    infers the protected attributes.
        This can be used to enforce fairness when no information about protected attribtutes is
        avalible at test time. If this is not false, fairness will be measured using the variable
        'groups', but enforced using the predictor response.
    use_fast: (Optional, default True) Bool or 'hybrid'
        If use_fast is True, the fair search is much more efficient, but the objectives must take the
        form of a GroupMetric. 'Hybrid' initialises the slow_pathway with the output of the fast pathhway
        This is useful for infered groups in deep networks.
        If use_fast is False, autogluon and scikitlearn scorers are also supported.
    conditioning_factor (optional, default None) Used to specify the factor conditional metrics
        are conditioned on.
        Takes the same form as groups.
    threshold (optional, default 2/3) used in use_fast pathway. Adds an extra catagory of uncertain
        group when infering attributes.
        If a datapoint has no response from the inferred_groups classifier above the threshold
        then it is assigned to the uncertain group. Tuning this value may help improve
        fairness/performance trade-offs.
        When set to 0 it is off.
    """

    def __init__(self, predictor, validation_data, groups=None, *, inferred_groups=False,
                 add_noise=False, logit_scaling=False,
                 use_fast=True, conditioning_factor=None, threshold=2/3) -> None:
        if predictor is None:
            def predictor(x):
                return x
            if isinstance(validation_data, dict):
                assert validation_data["data"].ndim == 2, "If predictor is None, validation_data['data'] must be a 2D array of predicted probabilities"
            else:
                assert validation_data.ndim == 2, "If predictor is None, validation_data must be a 2D array of predicted probabilities"
        if not is_not_autogluon(predictor) and predictor.problem_type != 'binary':
            logger.error('Fairpredictor only takes a binary predictor as input')

        assert use_fast == 'hybrid' or use_fast is False or use_fast is True
        # Check if sklearn
        _guard_predictor_data_match(validation_data, predictor)
        self.predictor = predictor
        if groups is None:
            if isinstance(validation_data, dict):
                if isinstance(validation_data.get('groups', None), str):
                    groups = validation_data['groups']
            else:
                groups = False
        else:
            if isinstance(validation_data, dict) and validation_data.get('groups', None) is not None:
                logger.warning(("Groups passed twice to fairpredictor both as part of "
                                "the dataset and as an argument. "
                                "The argument will be used."))
        # Internal logic differentiates between groups should be recovered from other data
        # i.e. groups = None
        # and there are no groups i.e. groups = False
        # However, as a user interface groups = None makes more sense for instantiation.
        # This colides with the alternative use case where a dict is passed and groups
        # are estimated.
        if use_fast == 'hybrid':
            self.threshold = 0
        else:
            self.threshold = threshold

        self.groups = groups
        self.use_fast: bool = use_fast
        self.conditioning_factor = conditioning_factor
        self.logit_scaling = logit_scaling
        if isinstance(validation_data, dict):
            self.validation_data = validation_data  # ['data']
            validation_labels = validation_data['target']
            if groups is None:
                groups = validation_data.get('groups', False)
                # Do not update self.groups otherwise this will stick
            if conditioning_factor is None:
                conditioning_factor = validation_data.get('cond_fact', False)
                # Do not update self.conditioning otherwise this will stick
            else:
                if validation_data.get('cond_fact', False) is not False:
                    logger.warning(("Conditioning factor passed twice to fairpredictor both as "
                                    "part of the dataset and as an argument. "
                                    "The argument will be used."))
        else:
            self.validation_data = validation_data
            validation_labels = self.validation_data[predictor.label]

        # We use _internal_groups as a standardized argument that is always safe to pass
        # to functions expecting a vector
        self._internal_groups = self.groups_to_numpy(groups, self.validation_data)
        self._internal_conditioning_factor = self.cond_fact_to_numpy(conditioning_factor, self.validation_data)
        assert self._internal_groups.shape[0] == validation_labels.shape[0], 'The size of the groups does not match the dataset size'
        assert np.unique(validation_labels).shape[0] <= 2, 'More than two target labels used. OxonFair only works with binary predictors'

        self.inferred_groups = inferred_groups
        if inferred_groups:
            self._val_thresholds = call_or_get_proba(inferred_groups, self.validation_data)
        else:
            # Use OneHot and store encoder so it will work on new data
            self.group_encoder: OneHotEncoder = OneHotEncoder(handle_unknown='ignore')
            self.group_encoder.fit(self._internal_groups.reshape(-1, 1))
            self._val_thresholds = self.group_encoder.transform(
                self._internal_groups.reshape(-1, 1)).toarray()
        self.proba = call_or_get_proba(predictor, self.validation_data)
        self.add_noise = add_noise
        if add_noise:
            self.proba += np.random.normal(0, add_noise, self.proba.shape)
        if is_not_autogluon(self.predictor):
            self.y_true = np.asarray(validation_labels)
        else:
            self.y_true = np.asarray(validation_labels == self.predictor.class_labels[1])
        self.frontier: Optional[Tuple] = None
        if self.use_fast is True:
            self.offset = np.zeros((self._val_thresholds.shape[1],))
        else:
            self.offset = np.zeros((self._val_thresholds.shape[1], self.proba.shape[1]))
        self.objective1 = None
        self.objective2 = None
        self.round = False

    def _to_numpy(self, x, data, name='groups', none_replace=None) -> Optional[np.ndarray]:
        """helper function for transforming groups into a numpy array of unique values

        Parameters
        ----------
        x: a standard represenations such as might be used for groups (see class doc)
        data: a pandas dataframe or a dict containing data
        name: optional string, the field extracted from data.
        none_replace: Default value used when nothing else is found.

        Returns
        -------
        numpy array
        """
        if data is None:
            data = self.validation_data
        if x is None and isinstance(none_replace, str):
            x = none_replace
        if x is None and isinstance(data, dict):
            x = data.get(name, None)
        if x is None:
            x = none_replace

        if isinstance(data, dict):
            data = data['data']
        if x is False:
            return np.zeros(data.shape[0])

        if callable(x):
            return self.infered_to_hard(x(data))
        if isinstance(x, str):
            return np.asarray(data[x])
        if isinstance(x, int):
            return np.asarray(data[:, x])
        if x is None:
            return None
        return np.asarray(x)

    def groups_to_numpy(self, groups, data):
        """helper function for transforming groups into a numpy array of unique values

        Parameters
        ----------
        groups: a standard represenations of groups (see class doc)
        data: a pandas dataframe, numpy array, or a dict containing data

        Returns
        -------
        numpy array
        """
        return self._to_numpy(groups, data, 'groups', self.groups)

    def cond_fact_to_numpy(self, fact, data):
        """helper function for transforming fact into a numpy array of unique values

        Parameters
        ----------
        fact: one of the standard represenations of conditioning factor
        data: a pandas dataframe, numpy array, or a dict containing data

        Returns
        -------
        numpy array
        """
        out = self._to_numpy(fact, data, 'cond_fact', self.conditioning_factor)
        # conditioning factor may be arbitary unique values like groups,
        if out is None:
            return out
        from sklearn.preprocessing import OrdinalEncoder
        encoder = OrdinalEncoder()
        out = encoder.fit_transform(out.reshape(-1, 1))
        return out.reshape(-1)

    def infered_to_hard(self, infered):
        "Map the output of infered groups into a hard assignment for use in the fast pathway"
        if self.inferred_groups is False or self.threshold == 0:
            return infered.argmax(1)

        drop = infered.max(1) < self.threshold
        out = infered.argmax(1)+1
        out[drop] = 0
        return out

    def fit(self, objective, constraint=group_metrics.accuracy, value=0.0, *,
            greater_is_better_obj=None, greater_is_better_const=None,
            recompute=True, tol=False, grid_width=False, threshold=None,
            additional_constraints=(), force_levelling_up=False):
        """Fits the chosen predictor to optimize an objective while satisfing a constraint.

        Parameters
        ----------
        objective: a BaseGroupMetric or Scorable to be optimised
        constraint (optional): a BaseGroupMetric or Scorable that must be above/below a certain
        value
        value (optional): float the value constraint must be above or below
        If neither constraint nor value are provided fit enforces the constraint that accuracy
        is greater or equal to zero.

        greater_is_better_obj: bool or None Governs if the objective is maximised (True) or
                             minimized (False).
                If None the value of objective.greater_is_better is used.
        greater_is_better_const: bool or None Governs if the constraint has to be greater (True) or
                                smaller (False) than value.
                If None the value of constraint.greater_is_better is used.
        recompute: governs if the the parato frontier should be recomputed. Use False to efficiently
                    adjusting the threshold while keeping objective and constraint fixed.
        tol: float or False. Can round the solutions found by predict_proba to within a particular
                            tolerance to prevent overfitting.
                               Generally not needed.
        grid_width: allows manual specification of the grid size. N.B. the overall computational
                    budget is O(grid_width**groups)
                 By default the grid_size is 30
        threshold: A float between 0 and 1 or None. If threshold is not None, this overwrites
                    the threshold used for assignment to a "don't know class" in the hard assignment
                    of inferened groups.
        additional_constraints: A list of tupples of the form (metric, value, direction[optional]).
        This will drop all solutions from the pareto frontier that do not satisfy all additional constraints
        force_levelling_up: None (default), +1, or -1.
                            If none, do nothing.
                            If +1 force all weights found to be non-negative  -- i.e. fit can only increase the selection rate.
                            If -1 force all weights found to be non-positive  -- i.e. fit can only decrease the selection rate.

        Returns
        -------
        Nothing
        """
        if threshold is not None:
            self.threshold = threshold
        if greater_is_better_obj is None:
            greater_is_better_obj = objective.greater_is_better
        if greater_is_better_const is None:
            greater_is_better_const = constraint.greater_is_better

        if recompute is True or self.frontier is None:
            self.compute_frontier(objective, constraint,
                                  greater_is_better_obj1=greater_is_better_obj,
                                  greater_is_better_obj2=greater_is_better_const, tol=tol,
                                  grid_width=grid_width,
                                  additional_constraints=additional_constraints,
                                  force_levelling_up=force_levelling_up)
        if greater_is_better_const:
            mask = self.frontier[0][1] >= value
        else:
            mask = self.frontier[0][1] <= value

        if mask.sum() == 0:
            logger.warning("No solutions satisfy the constraint found, selecting the" +
                           " closest solution.")
            weights = self.frontier[1]
            vmax = [self.frontier[0][1].argmin(),
                    self.frontier[0][1].argmax()][int(greater_is_better_const)]
        else:
            values = self.frontier[0][0][mask]
            weights = self.frontier[1].T[mask].T

            vmax = [values.argmin(),
                    values.argmax()][int(greater_is_better_obj)]
        self.offset = weights.T[vmax].T
        return self

    def compute_frontier(self, objective1, objective2, greater_is_better_obj1,
                         greater_is_better_obj2, *, tol=False,
                         grid_width=False, additional_constraints=(),
                         force_levelling_up=False) -> None:
        """ Computes the parato frontier. Internal logic used by fit

        Parameters
        ----------
        objective1: a BaseGroupMetric or Scorable to be optimised
        objective2: a BaseGroupMetric or Scorable to be optimised
        greater_is_better_obj1: bool or None Governs if the objective is maximised (True)
                                 or  minimized (False).
                If None the value of objective.greater_is_better is used.
        greater_is_better_obj2: bool or None Governs if the constraint has to be greater (True)
                                or  smaller (False) than value.
                If None the value of constraint.greater_is_better is used.
        tol: float or False. Can round the solutions found by predict_proba to within a given
                            tolerance to prevent overfitting
                            Generally not needed.
        grid_width: allows manual specification of the grid size. N.B. the overall computational
                    budget is O(grid_width**groups)

        Returns
        -------
        Nothing
        """
        self.objective1 = objective1
        self.objective2 = objective2
        objectives = (objective1, objective2) + tuple((a[0] for a in additional_constraints))

        direction = np.ones(2 + len(additional_constraints))
        values = np.ones(len(additional_constraints))
        if greater_is_better_obj1 is False:
            direction[0] = -1
        if greater_is_better_obj2 is False:
            direction[1] = -1

        for i, c in enumerate(additional_constraints):
            assert 3 >= len(c) >= 2
            if len(c) == 2:
                if c[0].greater_is_better is False:
                    direction[2+i] = -1
            else:
                assert c[2] == '<' or c[2] == '>'
                if c[2] == '<':
                    direction[2+i] = -1
            values[i] = c[1]

        proba = self.proba

        self.round = tol

        if tol is not False:
            proba = np.around(self.proba / tol) * tol
        proba = proba[:, 0] - proba[:, 1]

        if force_levelling_up:  # Truncate search space
            if force_levelling_up == '-':
                proba = np.minimum(proba,  -1e-6)
            else:
                proba = np.maximum(proba, 0)
        assert force_levelling_up in [False, '+', '-', True], 'force_levelling_up must be one of False, +, - or True'

        def call_slow(existing_weights=None):
            fix_obj = [fix_groups_and_conditioning(obj, self._internal_groups,
                                                   self._internal_conditioning_factor, self.y_true) for obj in objectives]
            if grid_width is False:
                gw = 18
            else:
                gw = grid_width
            coarse_thresh = self._val_thresholds  # np.asarray(self._val_thresholds, dtype=np.float16)
            return fair_frontier.build_coarse_to_fine_front(fix_obj,
                                                            self.y_true, proba,
                                                            coarse_thresh,
                                                            directions=direction,
                                                            nr_of_recursive_calls=3,
                                                            initial_divisions=gw,
                                                            logit_scaling=self.logit_scaling,
                                                            existing_weights=existing_weights,
                                                            additional_constraints=values)  # force_levelling_up=True)

        def call_fast(grid_width=grid_width):
            if grid_width is False:
                grid_width = min(30, (30**5)**(1 / self._val_thresholds.shape[1]))
            return efficient_compute.grid_search(self.y_true, proba, objectives,
                                                 self.infered_to_hard(self._val_thresholds),
                                                 self._internal_groups,
                                                 steps=min(30, (30**5)**(1 / self._val_thresholds.shape[1])),
                                                 directions=direction, factor=self._internal_conditioning_factor,
                                                 additional_constraints=values,
                                                 force_levelling_up=force_levelling_up)

        if self.use_fast == 'hybrid':
            frontier = call_fast(min(grid_width, min(30, (30**5)**(1 / self._val_thresholds.shape[1]))))
            weights = frontier[1]
            new_weights = np.zeros((weights.shape[0], 2, weights.shape[1]), dtype=weights.dtype)
            new_weights[::-1, 0, :] = weights
            self.frontier = call_slow(new_weights)
        elif self.use_fast is True:
            self.frontier = call_fast()
        else:
            self.frontier = call_slow()

    def frontier_thresholds(self):
        "Returns the thresholds corresponding to the found frontier"
        assert self.frontier, "Call fit before frontier_thresholds"
        return self.frontier[1]

    def frontier_scores(self, data=None):
        "Returns the scores (i.e. objective and constraint) corresponding to the found frontier"
        assert self.frontier, "Call fit before frontier_scores"
        if data is None:
            return self.frontier[0]

        objective1 = self.objective1
        objective2 = self.objective2

        if isinstance(data, dict):
            labels = np.asarray(data['target'])
            proba = call_or_get_proba(self.predictor, data['data'])

        else:
            assert not is_not_autogluon(self.predictor), 'Data must be a dict unless using autogluon predictors'
            labels = np.asarray(data[self.predictor.label])
            proba = call_or_get_proba(self.predictor, data)
            labels = (labels == self.predictor.positive_class) * 1
        if self.add_noise:
            proba += np.random.normal(0, self.add_noise, proba.shape)

        groups = self.groups_to_numpy(None, data)
        if groups is None:
            groups = np.ones_like(labels)

        if self.inferred_groups is False:
            if self.groups is False:
                val_thresholds = np.ones((groups.shape[0], 1))
            else:
                val_thresholds = self.group_encoder.transform(groups.reshape(-1, 1)).toarray()
        else:
            if isinstance(data, dict):
                val_thresholds = call_or_get_proba(self.inferred_groups, data['data'])
            else:
                val_thresholds = call_or_get_proba(self.inferred_groups, data)

        if self.use_fast is not True:
            factor = self._internal_conditioning_factor
            if _needs_groups(objective1):
                objective1 = fix_groups_and_conditioning(objective1,
                                                         self.groups_to_numpy(groups, data), factor, self.y_true)
            if _needs_groups(objective2):
                objective2 = fix_groups_and_conditioning(objective2,
                                                         self.groups_to_numpy(groups, data), factor, self.y_true)

            front1 = fair_frontier.compute_metric(objective1, labels, proba,
                                                  val_thresholds, self.frontier[1])
            front2 = fair_frontier.compute_metric(objective2, labels, proba,
                                                  val_thresholds, self.frontier[1])

        else:
            front1 = efficient_compute.compute_metric(objective1, labels, proba,
                                                      groups,
                                                      self.infered_to_hard(val_thresholds),
                                                      self.frontier[1])
            front2 = efficient_compute.compute_metric(objective2, labels, proba,
                                                      groups,
                                                      self.infered_to_hard(val_thresholds),
                                                      self.frontier[1])

        return (front1, front2)

    def set_threshold(self, threshold):
        """Set the thresholds.
           This code allows the manual overriding of the thresholds found by fit to enforce different trade-offs.
           """
        self.offset = threshold

    def plot_frontier(self, data=None, groups=None, *, objective1=False, objective2=False,
                      show_updated=True, show_original=True, color=None, new_plot=True, prefix='',
                      name_frontier='Frontier', subfig=None) -> None:
        """ Plots an existing parato frontier with respect to objective1 and objective2.
            These do not need to be the same objectives as used when computing the frontier
            The original predictor, and the predictor selected by fit is shown in different colors.
            fit() must be called first.

            Parameters
            ----------
            data: (optional) pandas dataset or dict. If not specified, uses the data used to run fit.
            groups: (optional) groups data (see class definition). If not specified, uses the
                                definition provided at initialisation
            objective1: (optional) an objective to be plotted, if not specified use the
                                    objective provided to fit is used in its place.
            objective2: (optional) an objective to be plotted, if not specified use the
                                    constraint provided to fit is used in its place.
            show_updated: (optional, default True) Highlight the updated classifier with a
                different marker
            color: (optional, default None) Specify the color the frontier should be plotted in.
            new_plot: (optional, default True) specifies if plt.figure() should be called at the
            start or if an existing plot should be overlayed
            prefix (optional string) an additional prefix string that will be added to the legend
            for frontier and updated predictor.
        """
        import matplotlib.pyplot as plt  # noqa: C0415
        assert self.frontier is not None, 'Call fit before plot_frontier.'
        _guard_predictor_data_match(data, self.predictor)
        objective1 = objective1 or self.objective1
        objective2 = objective2 or self.objective2
        if not subfig and new_plot:
            plt.figure()
        if subfig:
            ax = subfig
            ax.set_title('Frontier found')
            ax.set_ylabel(objective1.name)
            ax.set_xlabel(objective2.name)
        else:
            ax = plt
            ax.title('Frontier Found')
            ax.ylabel(objective1.name)
            ax.xlabel(objective2.name)

        if data is None:
            data = self.validation_data
            labels = self.y_true
            proba = self.proba
            groups = self.groups_to_numpy(groups, data)
            if groups is None:
                groups = np.ones_like(labels)
            val_thresholds = self._val_thresholds
        else:
            if isinstance(data, dict):
                labels = np.asarray(data['target'])
                proba = call_or_get_proba(self.predictor, data['data'])

            else:
                assert not is_not_autogluon(self.predictor), 'Data must be a dict unless using autogluon predictors'
                labels = np.asarray(data[self.predictor.label])
                proba = call_or_get_proba(self.predictor, data)
                labels = (labels == self.predictor.positive_class) * 1
            if self.add_noise:
                proba += np.random.normal(0, self.add_noise, proba.shape)

            groups = self.groups_to_numpy(groups, data)
            if groups is None:
                groups = np.ones_like(labels)

            if self.inferred_groups is False:
                if self.groups is False:
                    val_thresholds = np.ones((groups.shape[0], 1))
                else:
                    val_thresholds = self.group_encoder.transform(groups.reshape(-1, 1)).toarray()
            else:
                if isinstance(data, dict):
                    val_thresholds = call_or_get_proba(self.inferred_groups, data['data'])
                else:
                    val_thresholds = call_or_get_proba(self.inferred_groups, data)
        if self.use_fast is not True:
            factor = self._internal_conditioning_factor
            if _needs_groups(objective1):
                objective1 = fix_groups_and_conditioning(objective1,
                                                         self.groups_to_numpy(groups, data), factor, self.y_true)
            if _needs_groups(objective2):
                objective2 = fix_groups_and_conditioning(objective2,
                                                         self.groups_to_numpy(groups, data), factor, self.y_true)

            front1 = fair_frontier.compute_metric(objective1, labels, proba,
                                                  val_thresholds, self.frontier[1])
            front2 = fair_frontier.compute_metric(objective2, labels, proba,
                                                  val_thresholds, self.frontier[1])

            zero = [dispatch_metric(objective1, labels, proba, groups, factor),
                    dispatch_metric(objective2, labels, proba, groups, factor)]

            front1_u = fair_frontier.compute_metric(objective1, labels, proba,
                                                    val_thresholds, self.offset[:, :, np.newaxis])
            front2_u = fair_frontier.compute_metric(objective2, labels, proba,
                                                    val_thresholds, self.offset[:, :, np.newaxis])

        else:
            front1 = efficient_compute.compute_metric(objective1, labels, proba,
                                                      groups,
                                                      self.infered_to_hard(val_thresholds),
                                                      self.frontier[1])
            front2 = efficient_compute.compute_metric(objective2, labels, proba,
                                                      groups,
                                                      self.infered_to_hard(val_thresholds),
                                                      self.frontier[1])

            zero = [objective1(labels, proba.argmax(1), groups),
                    objective2(labels, proba.argmax(1), groups)]

            front1_u = efficient_compute.compute_metric(objective1, labels, proba, groups,
                                                        self.infered_to_hard(val_thresholds),
                                                        self.offset[:, np.newaxis])
            front2_u = efficient_compute.compute_metric(objective2, labels, proba, groups,
                                                        self.infered_to_hard(val_thresholds),
                                                        self.offset[:, np.newaxis])

        def cross_scatter(front1, front2, color=None, s=None, marker=None, label="", edgecolors=None):
            if not isinstance(front2, np.ndarray):
                front2 = np.asarray(front2).reshape(-1)
            if not isinstance(front1, np.ndarray):
                front1 = np.asarray(front1).reshape(-1)

            front2 = front2.reshape(front2.shape[0], -1)
            front1 = front1.reshape(front1.shape[0], -1)
            for i in range(max(front1.shape[-1], front2.shape[-1])):
                if front1.shape[-1] == 1:
                    i1 = 0
                else:
                    i1 = i
                if front2.shape[-1] == 1:
                    i2 = 0
                else:
                    i2 = i

                ax.scatter(front2[:, i2], front1[:, i1], label=label, c=color, s=s, marker=marker, edgecolors=edgecolors)
        cross_scatter(front1, front2, color=color, label=prefix+name_frontier)
        if show_original:
            cross_scatter(zero[0], zero[1], s=40, label='Original predictor', marker='*', edgecolors='k')
        if show_updated:
            cross_scatter(front1_u, front2_u, s=40, label=prefix+'Updated predictor', marker='s', edgecolors='k')
        ax.legend(loc='best')

    def evaluate(self, data=None, metrics=None, verbose=True) -> pd.DataFrame:
        """Compute standard metrics of the original predictor and the updated predictor
         found by fit and return them in a dataframe.
          If fit has not been called only return the metrics of the original predictor.

        Parameters
        ----------
        data: (optional) a pandas dataframe to evaluate over. If not provided evaluate over
            the dataset provided at initialisation.
        metrics: (optional) a dictionary where the keys are metric names and the elements are either
                    scoreables or group metrics. If not provided report the standard metrics
                    reported by autogluon on binary predictors

        Returns
        -------
        a pandas dataset containing rows indexed by metric name, and columns by
        ['original', 'updated']
         """
        _guard_predictor_data_match(data, self.predictor)
        if metrics is None:
            metrics = group_metrics.ag_metrics
        return self.evaluate_fairness(data, metrics=metrics, verbose=verbose)

    def evaluate_fairness(self, data=None, groups=None, factor=None, *,
                          metrics=None, verbose=True) -> pd.DataFrame:
        """Compute standard fairness metrics for the orginal predictor and the new predictor
         found by fit. If fit has not been called return a dataframe containing
         only the metrics of the original predictor.
         parameters
        ----------
        data: (optional) a pandas dataframe to evaluate over. If not provided evaluate over
                the dataset provided at initialisation.
        groups (optional) a specification of the groups (see class defintion). If not provided use
                the defintion provided at init.
        metrics: (optional) a dictionary where the keys are metric names and the elements are either
                    scoreables or group metrics. If not provided report the standard metrics
                    reported by SageMaker Clarify
                    https://mkai.org/learn-how-amazon-sagemaker-clarify-helps-detect-bias
        returns
        -------
        a pandas dataset containing rows indexed by fairness measure name, and columns by
        ['original', 'updated']
         """
        _guard_predictor_data_match(data, self.predictor)
        factor = self.cond_fact_to_numpy(factor, data)
        if metrics is None:
            metrics = group_metrics.default_fairness_measures

        if data is None:
            data = self.validation_data
            labels = self.y_true
            y_pred_proba = call_or_get_proba(self.predictor, data)
        else:
            if isinstance(data, dict):
                labels = data['target']
                y_pred_proba = call_or_get_proba(self.predictor, data['data'])
            else:
                labels = np.asarray(data[self.predictor.label])
                y_pred_proba = call_or_get_proba(self.predictor, data)
                if not is_not_autogluon(self.predictor):
                    labels = (labels == self.predictor.positive_class) * 1
        groups = self.groups_to_numpy(groups, data)
        score = y_pred_proba[:, 1] - y_pred_proba[:, 0]
        collect = perf.evaluate_fairness(labels, score, groups, factor,
                                         metrics=metrics, verbose=verbose, threshold=0)

        if self.frontier is not None:
            y_pred_proba = np.asarray(self.predict_proba(data))
            score = (y_pred_proba[:, 1]-y_pred_proba[:, 0])/2
            new_pd = perf.evaluate_fairness(labels, score, groups, factor,
                                            metrics=metrics, verbose=verbose,
                                            threshold=0)

            collect = pd.concat([collect, new_pd], axis='columns')
            collect.columns = ['original', 'updated']
        else:
            collect = pd.concat([collect,], axis='columns')
            collect.columns = ['original']

        return collect

    def fairness_metrics(self, y_true: np.ndarray, proba, groups: np.ndarray,
                         metrics, factor, *, verbose=True) -> pd.DataFrame:
        """Helper function for evaluate_fairness
        Report fairness metrics that do not require additional information.

        Parameters
        ----------
        y_true: numpy array containing true binary labels of the dataset
        proba: numpy or pandas array containing the output of predict_proba
        groups: numpy array containing discrete group labelling
        metrics: a dictionary where keys are the names and values are either
        Scorable or a BaseGroupMetric.

        Returns
        -------
        a pandas dataframe of fairness metrics
        """
        values = np.zeros(len(metrics))
        names = []
        for i, k in enumerate(metrics.keys()):
            if verbose is False:
                names.append(k)
            else:
                names.append(metrics[k].name)
            values[i] = dispatch_metric(metrics[k], y_true, proba, groups, factor)

        return pd.DataFrame(values, index=names)

    def evaluate_groups(self, data=None, groups=None, metrics=None, fact=None, *,
                        return_original=True, verbose=True):
        """Evaluate standard metrics per group and returns dataframe.

        Parameters
        ----------
        data: (optional) a pandas dataframe to evaluate over. If not provided evaluate over
            the dataset provided at initialisation.
        groups (optional) a specification of the groups (see class defintion). If not provided
                use the defintion provided at init.
        metrics: (optional) a dictionary where the keys are metric names and the elements are either
                    scoreables or group metrics. If not provided report the standard autogluon
                    binary predictor evaluations plus measures of the size of each group and their
                    labels.
        return_original: (optional) bool.
                            If return_original is true, it returns a hierarchical dataframe
                            of the scores of the original classifier under key 'original'and the
                            scores of the updated classifier under key 'updated'.
                            If return_original is false it returns a dataframe of the scores of the
                            updated classifier only.

        Returns
        -------
        either a dict of pandas dataframes or a single pandas dataframe, depending on the value of
        return original.
        """
        _guard_predictor_data_match(data, self.predictor)
        if metrics is None:
            metrics = group_metrics.default_group_metrics
        if data is None:
            data = self.validation_data
            y_true = self.y_true
            new_pred_proba = np.asarray(self.predict_proba(data))
            if return_original:
                orig_pred_proba = np.asarray(call_or_get_proba(self.predictor, data))
        else:
            if isinstance(data, dict):
                y_true = data['target']
                new_pred_proba = np.asarray(self.predict_proba(data))
                if return_original:
                    orig_pred_proba = call_or_get_proba(self.predictor, data['data'])
            else:
                y_true = np.asarray(data[self.predictor.label])
                new_pred_proba = np.asarray(self.predict_proba(data))
                if return_original:
                    orig_pred_proba = np.asarray(call_or_get_proba(self.predictor, data))
                y_true = (y_true == self.predictor.positive_class) * 1

        if self.add_noise and return_original:
            orig_pred_proba += np.random.normal(0, self.add_noise, orig_pred_proba.shape)

        groups = self.groups_to_numpy(groups, data)
        fact = self.cond_fact_to_numpy(fact, data)
        if groups is None:
            groups = np.ones_like(y_true, dtype=int)
        if return_original:
            score = orig_pred_proba[:, 1] - orig_pred_proba[:, 0]
            original = perf.evaluate_per_group(y_true, score, groups,
                                               fact,
                                               threshold=0,
                                               metrics=metrics,
                                               verbose=verbose)

        score = new_pred_proba[:, 1] - new_pred_proba[:, 0]
        updated = perf.evaluate_per_group(y_true, score, groups,
                                          fact,
                                          threshold=0,
                                          metrics=metrics,
                                          verbose=verbose)

        out = updated
        if self.frontier is None:
            out = pd.concat([updated, ], keys=['original', ])
        elif return_original:
            out = pd.concat([original, updated], keys=['original', 'updated'])
        return out

    def predict_proba(self, data, *, transform_features=True, force_normalization=False):
        """Duplicates the functionality of predictor.predict_proba for fairpredictor.

        Parameters
        ----------
        data a numpy/pandas array to make predictions over.

        Returns
        ------
        a  pandas array of scores. Note, these scores are not probabilities, and not guarenteed to
        be non-negative or to sum to 1.

        To make them positive and sum to 1 use force_normalization=True
        """
        data = copy.deepcopy(data)
        if self.groups is None and self.inferred_groups is False:
            _guard_predictor_data_match(data, self.predictor)
        if self.groups is None and isinstance(data, dict):
            groups = data.get('groups', False)
        else:
            groups = self.groups
        if isinstance(data, dict):
            data = data['data']

        if is_not_autogluon(self.predictor):
            proba = call_or_get_proba(self.predictor, data)
        else:
            proba: pd.DataFrame = self.predictor.predict_proba(data,
                                                               transform_features=transform_features)
        if self.add_noise:
            proba += np.random.normal(0, self.add_noise, proba.shape)

        if self.inferred_groups is False:
            if groups is False:
                onehot = np.ones((data.shape[0], 1))
            else:
                groups = self.groups_to_numpy(groups, data)
                onehot = self.group_encoder.transform(groups.reshape(-1, 1)).toarray()
        else:
            if isinstance(data, dict):
                onehot = call_or_get_proba(self.inferred_groups, data['data'])
            else:
                onehot = call_or_get_proba(self.inferred_groups, data)
        if self.use_fast is True:
            tmp = np.zeros_like(proba)
            cache = self.offset[self.infered_to_hard(onehot)]
            if force_normalization:
                tmp[:, 1] = np.maximum(cache, 0)
                tmp[:, 0] = np.maximum(-cache, 0)
            else:
                tmp[:, 1] = cache
        else:
            tmp2 = onehot.dot(self.offset)
            if force_normalization:
                tmp = np.zeros_like(proba)
                tmp[:, 1] = np.maximum(tmp2[:, 1] - tmp2[:, 0], 0)
                tmp[:, 0] = np.maximum(tmp2[:, 0] - tmp2[:, 1], 0)
            else:
                tmp = tmp2
        if self.round is not False:
            proba = np.around(proba / self.round) * self.round
        proba += tmp
        if force_normalization:
            sum = proba.sum(1)
            if isinstance(proba, pd.DataFrame):
                proba[proba.columns[0]] /= sum
                proba[proba.columns[1]] /= sum
            else:
                proba /= sum[:, np.newaxis]

        return proba

    def predict(self, data, *, transform_features=True) -> pd.Series:
        "duplicates the functionality of predictor.predict for fairpredictor"
        proba = self.predict_proba(data, transform_features=transform_features)
        if isinstance(proba, pd.DataFrame):
            return proba.idxmax(1)
        return np.argmax(proba, 1)

    def extract_coefficients(self):
        """Extracts coefficients used to combine the heads when creating a fair deep classifier.

        This code assumes only two groups and that second head of the model is trained to output single
        values with target values 0 and 1 corresponding to membership of one of two protected groups.

        If instead the second head returns  a 1-hot encoding, indicating membership of 2 or more groups,
        use extract_coefficients_1_hot.

        This code does not support objects created with use_fast=True.

        Returns
        -------
        1. a scalar a, and
        2. bias term b.

        Such that head_1 + a * head_2 + b has the same outputs as our fair classifier.
        This can be used to merge the coefficients of the two heads, creating a single-headed fair classifier.
        """
        return self.offset[1, 0]-self.offset[0, 0], -self.offset[1, 0]

    def extract_coefficients_1_hot(self):
        """Extracts coefficients used to combine the heads when creating a fair deep classifier.

        This code assumes that second head of the model is trained to output a one hot encoding
        corresponding to membership of a protected group.
        For more compact binary encodings see extract_coeefficents

        This code does not support objects created with use_fast=True.

        Returns
        -------
        A vector coefficient a.

        Such that head_1 + a.dot(head_2) has the same outputs as our fair classifier.
        This can be used to merge the coefficients of the two heads, creating a single-headed fair classifier.
        """
        return -self.offset[:, 0]

    def merge_heads_pytorch(self, heads):
        """Merges multiple heads into a single head of the same form, that enforces fairness.

        Parameters
        ----------

        heads: a 2-d torch linear layer of dimension: backbone width by number of heads.
        The first head is assumed to be the classifier response, and the remainder of heads encode the attributes.

        If the number of heads is two we asumme the second-head was trained to enocde a binary attributes with labels roughly 0 and 1.

        If the number of heads is more than two we assume all heads except the first encode an approximate 1-hot embedding of the attributes

        Returns
        --------
        A new linear head of size backbone width x 1 """
        from torch.nn import Linear
        from torch import Tensor
        assert isinstance(heads, Linear)
        assert heads.out_features > 1
        out = Linear(heads.in_features, 1, dtype=heads.weight.dtype)
        if heads.out_features == 2:
            assert self.offset.shape[0] == 2, 'Dimension mismatch between training data and heads'
            coeff = self.extract_coefficients()
            # Now we merge the weights
            out.weight.data[:] = (heads.weight[0] + coeff[0]*heads.weight[1]).data
            # and the biases
            out.bias.data[:] = (heads.bias[0] + coeff[0]*heads.bias[1]).data
            # and add the extra bias term
            out.bias.data += coeff[1]
        else:
            assert self.offset.shape[0] == heads.out_features-1, 'Dimension mismatch between training data and heads'
            coeff = Tensor(self.extract_coefficients_1_hot())
            out.weight.data[:] = (heads.weight[0] + coeff.inner(heads.weight[1:].T)).data
            out.bias.data[:] = (heads.bias[0] + coeff.dot(heads.bias[1:])).data
        return out


def _needs_groups(func) -> bool:
    """Internal helper function. Check if a metric is a scorer. If not assume it requires a group
    argument.

    Parameters
    ----------
    func either a Scorable or GroupMetric
    """
    if not AUTOGLUON_EXISTS:
        return True
    return not isinstance(func, Scorer)


def is_not_autogluon(predictor) -> bool:
    """Internal helper function. Checks if a predictor is not an autogluon tabular predictor.

    Parameters
    ----------
    predictor: some sklearn/autogluon like predictor """
    if AUTOGLUON_EXISTS:
        return not isinstance(predictor, TabularPredictor)
    return True


def call_or_get_proba(predictor, data) -> np.ndarray:
    """Internal helper function. Implicit dispatch depending on if predictor is callable
    or follows scikit-learn interface.
    Converts output to numpy array"""
    if isinstance(data, dict):
        data = data['data']
    if callable(predictor):
        out = np.asarray(predictor(data))
        if out.ndim == 1:
            width = out.max()+1
            new_out = np.zeros((out.shape[0], width))
            new_out[(np.arange(out.shape[0]), out)] = 1
            return new_out
        return out
    return np.asarray(predictor.predict_proba(data))


def _guard_predictor_data_match(data, predictor) -> None:
    """Internal helper function. Checks that data is in the right format."""
    if (data is not None
        and is_not_autogluon(predictor)
        and not (isinstance(data, dict) and
                 data.get('data', False) is not False and
                 data.get('target', False) is not False)):
        assert False, """When not using autogluon data must be a dict containing keys
                        'data' and 'target'"""


def inferred_attribute_builder(train, target, protected, *args, **kwargs):
    """Helper function that trains autogluon tabular predictors
    so fairness can be enforced without knowing the protected attribute at test time.

        Parameters
        ----------
        train: a pandas dataframe
        target: a string identifying the column of the dataframe the predictor should try to
        estimate.
        protected: a string identifying the column of the dataframe that represents the
        protected attribute.

        Returns
        -------
        a pair of autogluon tabular predictors.
            1. a predictor predicting the target that doesn't use the protected attribute
            2. a predictor predicting the protected attribute that doesn't use the target.

        """
    assert AUTOGLUON_EXISTS, 'Builder only works if autogluon is installed'
    target_train = train.drop(protected, axis=1, inplace=False)
    protected_train = train.drop(target, axis=1, inplace=False)
    target_predictor = TabularPredictor(label=target).fit(train_data=target_train, *args, **kwargs)
    protected_predictor = TabularPredictor(label=protected)
    protected_predictor.fit(train_data=protected_train, *args, **kwargs)
    return target_predictor, protected_predictor


def groups_to_masks(groups):
    "helper function to convert a sequence of groups to a 1-hot encoded set of masks"
    groups = np.asarray(groups)
    unique = np.unique(groups)
    mask = np.zeros((unique.shape[0], groups.shape[0]))
    for i, group_name in enumerate(unique):
        mask[i] = groups == group_name
    return mask.T


def fix_groups(metric, groups):
    """Fixes the choice of groups so that BaseGroupMetrics can be passed as Scorable analogs to the
    slow pathway.

    This substantially decreases runtime in the slow pathway.

    Parameters
    ----------
    metric: a BaseGroupMetric
    groups: a 1D pandas dataframe or numpy array

    Returns
    -------
    a function that takes y_true and y_pred as an input.

        todo: return scorable"""
    if (isinstance(metric, ScorerRequiresContPred) or
       (AUTOGLUON_EXISTS and isinstance(metric, Scorer) and (metric.needs_pred is False))):
        return metric

    groups = groups_to_masks(groups)

    if isinstance(metric, Overall):  # Performance hack. If metric is of type overall, groups don't matter -- assign all groups to 1.
        groups = np.ones(groups.shape[0])

    def new_metric(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return metric(y_true, y_pred, groups)
    return new_metric


def fix_conditioning(metric: BaseGroupMetric, conditioning_factor):
    """fixes the choice of groups so that BaseGroupMetrics can be passed as Scorable analogs to the
    slow pathway.

    Parameters
    ----------
    metric: a BaseGroupMetric
    groups: a 1D pandas dataframe or numpy array

    Returns
    -------
    a function that takes y_true and y_pred as an input.

        todo: return scorable"""
    if (isinstance(metric, ScorerRequiresContPred) or
       (AUTOGLUON_EXISTS and isinstance(metric, Scorer)) or metric.cond_weights is None):
        logger.warning("Fixing conditoning factor on a metric that doesn't use it.")
        return metric
    conditioning_factor = np.asarray(conditioning_factor)

    def new_metric(y_true: np.ndarray, y_pred: np.ndarray, groups) -> np.ndarray:
        return metric(y_true, y_pred, groups, conditioning_factor)
    return new_metric


def fix_groups_and_conditioning(metric, groups, conditioning_factor, y_true):
    """fixes the choice of groups and conditioning factor so that BaseGroupMetrics can be passed as
    Scorable analogs to the slow pathway.

    Parameters
    ----------
    metric: a BaseGroupMetric
    groups: a 1D pandas dataframe or numpy array

    Returns
    -------
    a function that takes y_true and y_pred as an input.

        todo: return scorable"""
    if (isinstance(metric, ScorerRequiresContPred) or
       (AUTOGLUON_EXISTS and isinstance(metric, Scorer)) or metric.cond_weights is None):
        return fix_groups(metric, groups)

    conditioning_factor = np.asarray(conditioning_factor)
    groups = np.asarray(groups)
    weights = metric.cond_weights(conditioning_factor, groups, y_true)
    groups = groups_to_masks(groups)

    if isinstance(metric, Overall):  # Performance hack. If metric is of type overall, groups don't matter -- assign all groups to 1.
        groups = np.ones(groups.shape[0])


    def new_metric(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return metric(y_true, y_pred, groups, weights)
    return new_metric


def dispatch_metric(metric, y_true, proba, groups, factor) -> float:
    """Helper function for making sure different types of Scorer and GroupMetrics get the right data

    Parameters
    ----------
    metric: a BaseGroupMetric or Scorable
    y_true: a binary numpy array indicating positive or negative labels
    proba: a 2xdatapoints numpy or pandas array
    groups: a numpy array indicating group membership.

    Returns
    -------
     a numpy array containing the score provided by metrics
    """
    proba = np.asarray(proba)
    try:
        if isinstance(metric, BaseGroupMetric):
            if metric.cond_weights is None:
                return metric(y_true, proba.argmax(1), groups)[0]
            return metric(y_true, proba.argmax(1), groups, factor)[0]

        if (AUTOGLUON_EXISTS and (isinstance(metric, Scorer) and (metric.needs_pred is False)) or
           isinstance(metric, group_metrics.ScorerRequiresContPred)):
            return metric(y_true, proba[:, 1] - proba[:, 0])

        return metric(y_true, proba.argmax(1))
    except ValueError:
        return np.nan


def single_threshold(x) -> np.ndarray:
    """A helper function. Allows you to measure and enforces fairness and performance measures
    by altering a single threshold for all groups.

    To use call FairPredictor with the argument infered_groups=single_threshold"""
    return np.ones((x.shape[0], 1))


def DataDict(target, data, groups=None, conditioning_factor=None) -> dict:
    "Helper function that builds dictionaries for use with sklearn classifiers"
    assert target.shape[0] == data.shape[0]
    assert data.ndim == 2
    assert target.ndim == 1 or (target.ndim == 2 and target.shape[1] == 1)
    target = np.asarray(target).reshape(-1)
    out = {'target': target, 'data': data}
    if groups is not None:
        if not isinstance(groups, str):
            assert target.shape[0] == groups.shape[0]
            assert groups.ndim == 1 or (groups.ndim == 2 and groups.shape[1] == 1)
            out['groups'] = np.asarray(groups).reshape(-1)
        else:
            out['groups'] = groups
    if conditioning_factor is not None:
        if not isinstance(conditioning_factor, str):
            assert conditioning_factor.ndim == 1
            assert target.shape[0] == conditioning_factor.shape[0]
        out['cond_fact'] = conditioning_factor
    return out


def DeepDataDict(target, score, groups, groups_inferred=None, *,
                 conditioning_factor=None) -> dict:
    """Wrapper around DataDict for deeplearning with inferred attributes.
     It transforms the input data into a dict, and creates helper functions so
     fairpredictor treats them appropriately.

     Parameters
     ----------
     target: a numpy array containing the values the classifier should predict(AKA groundtruth)
     score: a numpy array that is either size n by 1, and contains a logit output or n by (1 + #groups)
     and is a concatination of the logit output with the inferered groups.
     groups: a numpy array containing true group membership.
     infered_groups: optional numpy array of size n by #groups. If score is n by 1, infered groups go here.

     Returns
     -------
     A dict that can be passed to fairpredictor
    """
    assert score.ndim == 2
    assert target.ndim == 1
    assert groups.ndim == 1
    assert score.shape[0] == target.shape[0]
    assert target.shape[0] == groups.shape[0]
    assert score.shape[1] >= 1
    if groups_inferred is not None:
        assert score.shape[1] == 1
        assert groups_inferred.ndim == 2
        assert target.shape[0] == groups_inferred.shape[0]
        data = np.stack((score, groups_inferred), 1)
    else:
        # assert score.shape[1] > 1, 'When groups_inferred is None, score must also contain inferred group information'
        data = score
    return DataDict(target, data, groups, conditioning_factor=conditioning_factor)


def DeepFairPredictor(target, score, groups, groups_inferred=None,
                      *, conditioning_factor=None, truncate_logits=15,
                      use_actual_groups=False, use_fast=None,
                      logit_scaling=False) -> FairPredictor:
    """Wrapper around FairPredictor for deeplearning with inferred attributes.
     It transforms the input data into a dict, and creates helper functions so
     fairpredictor treats them appropriately.

     Paramters
     ---------

     target: a numpy array containing the values the classifier should predict(AKA groundtruth)
     score: a numpy array that is either size n by 1, and contains a logit output or n by (1 + #groups)
            and is a concatination of the logit output with the inferered groups.
     groups: a numpy array containing true group membership.
     infered_groups: optional numpy array of size n by #groups. If score is n by 1, infered groups go here.
     truncated_logits: for performance reasons we truncate the logits to lie in [-15,15] by default. Change this here.
     use_actual_groups: bool or 'single_threshold'. Indicates if we should use actual groups, inferred groups,
                or a single global threshold for all datapoints, to enforce fairness.
     use_fast: True, False or 'hybrid' (hybrid is prefered for infered groups. Initialises the slow pathway
            with the output of the fast pathway). By default 'hybrid' unless use_actual_groups is true, in which
            case True

    Returns
    -------
    A fairpredictor
     """
    val_data = DeepDataDict(target, score, groups, groups_inferred, conditioning_factor=conditioning_factor)

    def square_align(array):
        return np.stack((array[:, 1], 1-array[:, 1]), 1)

    def mult_group(array):
        return array[:, 1:]

    if groups_inferred:
        if groups_inferred.shape[1] == 1:
            group_fn = square_align
        else:
            group_fn = mult_group
    else:
        if score.shape[1] == 2:
            group_fn = square_align
        else:
            group_fn = mult_group

    def capped_identity(array):
        array = np.minimum(array[:, 0], truncate_logits)
        array = np.maximum(array, -truncate_logits)
        return np.stack((-array / 2, array / 2), 1)

    if use_fast is None:
        if use_actual_groups:
            use_fast = True
        else:
            use_fast = 'hybrid'
    if use_actual_groups is True:
        fpred = FairPredictor(capped_identity, val_data, threshold=0, use_fast=use_fast, logit_scaling=logit_scaling)
    elif use_actual_groups == 'single_threshold':
        fpred = FairPredictor(capped_identity, val_data, inferred_groups=single_threshold,
                              threshold=0, use_fast=use_fast, logit_scaling=False)
    else:
        fpred = FairPredictor(capped_identity, val_data, inferred_groups=group_fn, threshold=0,
                              use_fast=use_fast, logit_scaling=logit_scaling)
    return fpred
