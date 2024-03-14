"""The entry point to fair. Defines the FairPredictor object used to access fairness functionality."""
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from ..utils import group_metrics
from ..utils.group_metric_classes import BaseGroupMetric
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
    """Assess and mitigate the unfairness and effectiveness of a autogluon binary predictor post-fit
    by computing group specific metrics, and performing threshold adjustment.
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
    use_fast: (Optional) Bool
    If use_fast is True, the fair search is much more efficient, but the objectives must take the
    form of a GroupMetric
    If use_fast is False, autogluon scorers are also supported.
    conditioning_factor (optional, default None) Used to specify the factor conditional metrics
    are conditioned on.
    Takes the same form as groups.
    Threshold (optional, default 2/3) used in use_fast pathway. Adds an extra catagory of uncertain
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
        if not is_not_autogluon(predictor) and predictor.problem_type != 'binary':
            logger.error('Fairpredictor only takes a binary predictor as input')

        # Check if sklearn
        _guard_predictor_data_match(validation_data, predictor)
        self.predictor = predictor
        if groups is None:
            groups = False
        # Internal logic differentiates between groups should be recovered from other data
        # i.e. groups = None
        # and there are no groups i.e. groups = False
        # However, as a user interface groups = None makes more sense for instantiation.

        self.threshold = threshold
        self.groups = groups
        self.use_fast: bool = use_fast
        self.conditioning_factor = conditioning_factor
        self.logit_scaling = logit_scaling
        if isinstance(validation_data, dict):
            self.validation_data = validation_data['data']
            validation_labels = validation_data['target']
            if groups is False:
                groups = validation_data.get('groups', False)
                # Do not update self.groups otherwise this will stick
            else:
                if validation_data.get('groups', False) is not False:
                    logger.warning("""Groups passed twice to fairpredictor both as part of
                                   the dataset and as an argument.
                                   The argument will be used.""")
            if conditioning_factor is False:
                conditioning_factor = validation_data.get('cond_fact', False)
                # Do not update self.conditioning otherwise this will stick
            else:
                if validation_data.get('cond_fact', False) is not False:
                    logger.warning("""Conditioning factor passed twice to fairpredictor both as
                                   part of the dataset and as an argument.
                                   The argument will be used.""")
        else:
            self.validation_data = validation_data
            validation_labels = self.validation_data[predictor.label]

        # We use _internal_groups as a standardized argument that is always safe to pass
        # to functions expecting a vector
        self._internal_groups = self.groups_to_numpy(groups, self.validation_data)

        if self._internal_groups.shape[0] != validation_labels.shape[0]:
            logger.error('The size of the groups does not match the dataset size')

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
        self.frontier = None
        if self.use_fast:
            self.offset = np.zeros((self._val_thresholds.shape[1],))
        else:
            self.offset = np.zeros((self._val_thresholds.shape[1], self.proba.shape[1]))
        self.objective1 = None
        self.objective2 = None
        self.round = False

    def _to_numpy(self, groups, data, name='groups', none_replace=None) -> np.ndarray:
        """helper function for transforming groups into a numpy array of unique values
        parameters
        ----------
        groups: one of the standard represenations of groups (see class doc)
        data: a pandas dataframe or a dict containing data
        returns
        -------
        numpy array
        """
        if data is None:
            data = self.validation_data
        if groups is None and isinstance(data, dict):
            groups = data.get(name, None)
        if groups is None:
            groups = none_replace
        if isinstance(data, dict):
            data = data['data']
        if groups is False:
            return np.zeros(data.shape[0])

        if callable(groups):
            return self.infered_to_hard(groups(data))
        if isinstance(groups, str):
            return np.asarray(data[groups])
        if isinstance(groups, int):
            return np.asarray(data[:, groups])
        if groups is None:
            return None
        return np.asarray(groups)

    def groups_to_numpy(self, groups, data):
        """helper function for transforming groups into a numpy array of unique values
        parameters
        ----------
        groups: one of the standard represenations of groups (see class doc)
        data: a pandas dataframe, numpy array, or a dict containing data
        returns
        -------
        numpy array
        """
        return self._to_numpy(groups, data, 'groups', self.groups)

    def cond_fact_to_numpy(self, fact, data):
        """helper function for transforming fact into a numpy array of unique values
        parameters
        ----------
        fact: one of the standard represenations of conditioning factor
        data: a pandas dataframe, numpy array, or a dict containing data
        returns
        -------
        numpy array
        """
        return self._to_numpy(fact, data, 'cond_fact', self.conditioning_factor)

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
            recompute=True, tol=False, grid_width=False, threshold=None):
        """Fits the chosen predictor to optimize an objective while satisfing a constraint.
        parameters
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
        returns
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
                                  grid_width=grid_width)
        if greater_is_better_const:
            mask = self.frontier[0][1] >= value
        else:
            mask = self.frontier[0][1] <= value

        if mask.sum() == 0:
            logger.warning("""No solutions satisfy the constraint found, selecting the
                           closest solution""")
            weights = self.frontier[1]
            vmax = [self.frontier[0][1].argmin(),
                    self.frontier[0][1].argmax()][int(greater_is_better_const)]
        else:
            values = self.frontier[0][0][mask]
            weights = self.frontier[1].T[mask].T

            vmax = [values.argmin(),
                    values.argmax()][int(greater_is_better_obj)]
        self.offset = weights.T[vmax].T

    def compute_frontier(self, objective1, objective2, greater_is_better_obj1,
                         greater_is_better_obj2, *, tol=False,
                         grid_width=False) -> None:
        """ Computes the parato frontier. Internal logic used by fit
        parameters
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
        returns
        -------
        Nothing
        """
        self.objective1 = objective1
        self.objective2 = objective2

        if self.use_fast is False:
            factor = self.cond_fact_to_numpy(self.conditioning_factor, self.validation_data)
            if _needs_groups(objective1):
                objective1 = fix_groups_and_conditioning(objective1, self._internal_groups, factor)
            if _needs_groups(objective2):
                objective2 = fix_groups_and_conditioning(objective2, self._internal_groups, factor)
        direction = np.ones(2)
        if greater_is_better_obj1 is False:
            direction[0] = -1
        if greater_is_better_obj2 is False:
            direction[1] = -1

        if grid_width is False:
            if self.use_fast:
                grid_width = min(30, (30**5)**(1 / self._val_thresholds.shape[1]))
            else:
                grid_width = 14
                if self._val_thresholds.shape[1] == 2:
                    grid_width = 18

        self.round = tol

        proba = self.proba
        if tol is not False:
            proba = np.around(self.proba / tol) * tol
        if self.use_fast:
            fact = self.cond_fact_to_numpy(self.conditioning_factor, self.validation_data)
            self.frontier = efficient_compute.grid_search(self.y_true, proba, objective1,
                                                          objective2,
                                                          self.infered_to_hard(self._val_thresholds),
                                                          self._internal_groups, steps=grid_width,
                                                          directions=direction,
                                                          factor=fact)
        else:
            coarse_thresh = np.asarray(self._val_thresholds, dtype=np.float16)
            self.frontier = fair_frontier.build_coarse_to_fine_front(objective1, objective2,
                                                                     self.y_true, proba,
                                                                     coarse_thresh,
                                                                     directions=direction,
                                                                     nr_of_recursive_calls=3,
                                                                     initial_divisions=grid_width,
                                                                     logit_scaling=self.logit_scaling)

    def plot_frontier(self, data=None, groups=None, *, objective1=False, objective2=False,
                      show_updated=True, show_original=True, color=None, new_plot=True, prefix='',
                      name_frontier='Frontier') -> None:
        """ Plots an existing parato frontier with respect to objective1 and objective2.
            These do not need to be the same objectives as used when computing the frontier
            The original predictor, and the predictor selected by fit is shown in different colors.
            fit() must be called first.
            parameters
            ----------
            data: (optional) pandas dataset. If not specified, uses the data used to run fit.
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
        _guard_predictor_data_match(data, self.predictor)
        if self.frontier is None:
            logger.error('Call fit before plot_frontier')

        objective1 = objective1 or self.objective1
        objective2 = objective2 or self.objective2
        if new_plot:
            plt.figure()
        plt.title('Frontier found')
        plt.xlabel(objective2.name)
        plt.ylabel(objective1.name)

        if data is None:
            data = self.validation_data
            labels = self.y_true
            proba = self.proba
            groups = self.groups_to_numpy(groups, data)
            val_thresholds = self._val_thresholds
        else:
            if isinstance(data, dict):
                labels = np.asarray(data['target'])
                proba = call_or_get_proba(self.predictor, data['data'])

            else:
                labels = np.asarray(data[self.predictor.label])
                proba = call_or_get_proba(self.predictor, data)
                labels = (labels == self.predictor.positive_class) * 1
            if self.add_noise:
                proba += np.random.normal(0, self.add_noise, proba.shape)

            groups = self.groups_to_numpy(groups, data)

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
        if self.use_fast is False:
            factor = self.cond_fact_to_numpy(self.conditioning_factor, data)
            if _needs_groups(objective1):
                objective1 = fix_groups_and_conditioning(objective1,
                                                         self.groups_to_numpy(groups, data), factor)
            if _needs_groups(objective2):
                objective2 = fix_groups_and_conditioning(objective2,
                                                         self.groups_to_numpy(groups, data), factor)

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
        if color is None:
            plt.scatter(front2, front1, label=prefix+name_frontier)
            if show_original:
                plt.scatter(zero[1], zero[0], s=40, label='Original predictor', marker='*')
            if show_updated:
                plt.scatter(front2_u, front1_u, s=40, label=prefix+'Updated predictor', marker='s')
            plt.legend(loc='best')
        else:
            plt.scatter(front2, front1, c=color)

    def evaluate(self, data=None, metrics=None, verbose=True) -> pd.DataFrame:
        """Compute standard metrics of the original predictor and the updated predictor
         found by fit and return them in a dataframe.
          If fit has not been called only return the metrics of the original predictor.
        parameters
        ----------
        data: (optional) a pandas dataframe to evaluate over. If not provided evaluate over
            the dataset provided at initialisation.
        metrics: (optional) a dictionary where the keys are metric names and the elements are either
                    scoreables or group metrics. If not provided report the standard metrics
                    reported by autogluon on binary predictors
        returns
        -------
        a pandas dataset containing rows indexed by metric name, and columns by
        ['original', 'updated']
         """
        _guard_predictor_data_match(data, self.predictor)
        if metrics is None:
            metrics = group_metrics.ag_metrics
        groups = None
        if data is not None:
            if isinstance(data, dict):
                groups = np.ones(data['data'].shape[0])
            else:
                groups = np.ones(data.shape[0])
        else:
            groups = np.ones(self.validation_data.shape[0])

        return self.evaluate_fairness(data, groups, metrics=metrics, verbose=verbose)

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
            metrics = group_metrics.clarify_metrics

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
        collect.columns = ['original']

        if np.any(self.offset):
            y_pred_proba = np.asarray(self.predict_proba(data))
            score = y_pred_proba[:, 1]-y_pred_proba[:, 0]
            new_pd = perf.evaluate_fairness(labels, score, groups, factor,
                                            metrics=metrics, verbose=verbose,
                                            threshold=0)

            new_pd.columns = ['updated']
            collect = pd.concat([collect, new_pd], axis='columns')
        return collect

    def fairness_metrics(self, y_true: np.ndarray, proba, groups: np.ndarray,
                         metrics, factor, *, verbose=True) -> pd.DataFrame:
        """Helper function for evaluate_fairness
        Report fairness metrics that do not require additional information.
        parameters
        ----------
        y_true: numpy array containing true binary labels of the dataset
        proba: numpy or pandas array containing the output of predict_proba
        groups: numpy array containing discrete group labelling
        metrics: a dictionary where keys are the names and values are either
        Scorable or a BaseGroupMetric.
        returns
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
        parameters
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
        returns
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
        if return_original:
            out = pd.concat([original, updated], keys=['original', 'updated'])
        return out

    def predict_proba(self, data, *, transform_features=True):
        """Duplicates the functionality of predictor.predict_proba with the updated predictor.
        parameters
        ----------
        data a pandas array to make predictions over.
        return
        ------
        a  pandas array of scores. Note, these scores are not probabilities, and not guarenteed to
        be non-negative or to sum to 1.
        """
        if self.groups is False and isinstance(data, dict):
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
        if self.use_fast:
            tmp = np.zeros_like(proba)
            tmp[:, 1] = self.offset[self.infered_to_hard(onehot)]
        else:
            tmp = onehot.dot(self.offset)
        if self.round is not False:
            proba = np.around(proba / self.round) * self.round
        proba += tmp
        return proba

    def predict(self, data, *, transform_features=True) -> pd.Series:
        "duplicates the functionality of predictor.predict but with the fair predictor"
        proba = self.predict_proba(data, transform_features=transform_features)
        if isinstance(proba, pd.DataFrame):
            return proba.idxmax(1)
        return np.argmax(proba, 1)


def _needs_groups(func) -> bool:
    """Internal helper function. Check if a metric is a scorer. If not assume it requires a group
    argument.
    parameters
    ----------
    func either a Scorable or GroupMetric
    """
    if not AUTOGLUON_EXISTS:
        return True
    return not isinstance(func, Scorer)


def is_not_autogluon(predictor) -> bool:
    """Internal helper function. Checks if a predictor is not an autogluon fuction."""
    if AUTOGLUON_EXISTS:
        return not isinstance(predictor, TabularPredictor)
    return True


def call_or_get_proba(predictor, data):
    """Internal helper function. Implicit dispatch depending on if predictor is callable
    or follows scikit-learn interface.
    Converts output to numpy array"""
    if callable(predictor):
        return np.asarray(predictor(data))
    return np.asarray(predictor.predict_proba(data))


def _guard_predictor_data_match(data, predictor):
    if (data is not None
        and is_not_autogluon(predictor)
        and not (isinstance(data, dict) and
                 data.get('data', False) is not False and
                 data.get('target', False) is not False)):
        logger.error("""When not using autogluon data must be a dict containing keys
                        'data' and 'target'""")
        assert False


def inferred_attribute_builder(train, target, protected, *args, **kwargs):
    """Helper function that trains tabular predictors suitible for use when the protected attribute
        is inferred when enforcing fairness.
        parameters
        ----------
        train: a pandas dataframe
        target: a string identifying the column of the dataframe the predictor should try to
        estimate.
        protected: a string identifying the column of the dataframe that represents the
        protected attribute.
        returns
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


def fix_groups(metric: BaseGroupMetric, groups):
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
    groups = np.asarray(groups)

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
    if metric.cond_weights is None:
        logger.warning("Fixing conditoning factor on a metric that doesn't use it.")
        return metric
    conditioning_factor = np.asarray(conditioning_factor)

    def new_metric(y_true: np.ndarray, y_pred: np.ndarray, groups) -> np.ndarray:
        return metric(y_true, y_pred, groups, conditioning_factor)
    return new_metric


def fix_groups_and_conditioning(metric: BaseGroupMetric, groups, conditioning_factor):
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
    if metric.cond_weights is None:
        return fix_groups(metric, groups)

    conditioning_factor = np.asarray(conditioning_factor)
    groups = np.asarray(groups)

    def new_metric(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return metric(y_true, y_pred, groups, conditioning_factor)
    return new_metric


def dispatch_metric(metric: BaseGroupMetric, y_true, proba, groups, factor) -> np.ndarray:
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


def single_offset(x):
    """A helper function. Allows you to measure and enforces fairness and performance measures
    by altering a single threshold for all groups.
    To use call FairPredictor with the argument infered_groups=single_offset"""
    return np.zeros((x.shape[0], 1))
