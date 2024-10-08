"""Definitions of standard measures for fairness and performance"""

import numpy as np

from .scipy_metrics_cont_wrapper import roc_auc, average_precision, ScorerRequiresContPred  # noqa: F401
from .group_metric_classes import ( # pylint: disable=unused-import # noqa
    GroupMetric,
    AddGroupMetrics, MaxGroupMetrics,
    Utility)  # noqa: F401
# N.B. BaseGroupMetric and Utility are needed for type declarations


def ge1(x):
    """Helper function.
    Return the elementwise maximum of x or 1.
    Used so that metrics of the form 0/0 are treated as 0 not NaN."""
    return np.maximum(1, x)


# Basic parity measures for fairness
count = GroupMetric(lambda TP, FP, FN, TN: TP + FP + FN + TN, "Number of Datapoints")
pos_data_count = GroupMetric(lambda TP, FP, FN, TN: TP + FN, "Positive Count")
neg_data_count = GroupMetric(lambda TP, FP, FN, TN: FP + TN, "Negative Count")
pos_data_rate = GroupMetric(
    lambda TP, FP, FN, TN: (TP + FN) / (TP + FP + FN + TN), "Positive Label Rate"
)
neg_data_rate = GroupMetric(
    lambda TP, FP, FN, TN: (TN + FP) / (TP + FP + FN + TN), "Negative Label Rate"
)
pos_pred_rate = GroupMetric(
    lambda TP, FP, FN, TN: (TP + FP) / (TP + FP + FN + TN), "Positive Prediction Rate"
)
neg_pred_rate = GroupMetric(
    lambda TP, FP, FN, TN: (TN + FN) / (TP + FP + FN + TN), "Negative Prediction Rate"
)

# Standard metrics see sidebar of https://en.wikipedia.org/wiki/Precision_and_recall
true_pos_rate = GroupMetric(
    lambda TP, FP, FN, TN: (TP) / ge1(TP + FN), "True Positive Rate"
)
true_neg_rate = GroupMetric(
    lambda TP, FP, FN, TN: (TN) / ge1(FP + TN), "True Negative Rate"
)
false_pos_rate = GroupMetric(
    lambda TP, FP, FN, TN: (FP) / ge1(FP + TN), "False Positive Rate"
)
false_neg_rate = GroupMetric(
    lambda TP, FP, FN, TN: (FN) / ge1(TP + FN), "False Negative Rate"
)
pos_pred_val = GroupMetric(
    lambda TP, FP, FN, TN: (TP) / ge1(TP + FP), "Positive Predicted Value"
)
neg_pred_val = GroupMetric(
    lambda TP, FP, FN, TN: (TN) / ge1(TN + FN), "Negative Predicted Value"
)

# Existing binary metrics for autogluon
accuracy = GroupMetric(
    lambda TP, FP, FN, TN: (TP + TN) / (TP + FP + FN + TN), "Accuracy"
)
balanced_accuracy = GroupMetric(
    lambda TP, FP, FN, TN: (TP / ge1(TP + FN) + TN / ge1(TN + FP)) / 2,
    "Balanced Accuracy",
)
min_accuracy = GroupMetric(
    lambda TP, FP, FN, TN: np.minimum(TP / ge1(TP + FN), TN / ge1(TN + FP)),
    "Minimum-Label-Accuracy",
)  # common in min-max fairness literature
f1 = GroupMetric(lambda TP, FP, FN, TN: (2 * TP) / ge1(2 * TP + FP + FN), "F1 score")
precision = pos_pred_val.clone("Precision")
recall = true_pos_rate.clone("Recall")
mcc = GroupMetric(
    lambda TP, FP, FN, TN: (TP * TN - FP * FN)
    / ge1(np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))),
    "MCC",
)

bias_amplification = GroupMetric(lambda TP, FP, FN, TN: np.abs(FN - FP)/(TP + FP + FN + TN),
                                 'Absolute Bias Amplification', False)
# absolute metric based on the delta a_t term from directional bias amplification
# see: https://proceedings.mlr.press/v139/wang21t/wang21t.pdf

default_accuracy_metrics = {
    "accuracy": accuracy,
    "balanced_accuracy": balanced_accuracy,
    "f1": f1,
    "mcc": mcc,
}
additional_ag_metrics = {"precision": precision, "recall": recall, "roc_auc": roc_auc}
standard_metrics = {
    "true_pos_rate": true_pos_rate,
    "true_neg_rate": true_neg_rate,
    "false_pos_rate": false_pos_rate,
    "false_neg_rate": false_neg_rate,
    "pos_pred_val": pos_pred_val,
    "neg_pred_val": neg_pred_val,
}

ag_metrics = {**default_accuracy_metrics, **additional_ag_metrics}

count_metrics = {
    #   "count": count,
    "pos_data_count": pos_data_count,
    "neg_data_count": neg_data_count,
    "pos_data_rate": pos_data_rate,
    "pos_pred_rate": pos_pred_rate,
}
default_group_metrics = {**ag_metrics, **count_metrics}

extended_group_metrics = {
    **default_accuracy_metrics,
    **standard_metrics,
    **count_metrics,
}

# Postprocessing Clarify metrics
# https://mkai.org/learn-how-amazon-sagemaker-clarify-helps-detect-bias
class_imbalance = pos_data_rate.diff.clone("Class Imbalance")
demographic_parity = pos_pred_rate.diff.clone("Demographic Parity")
disparate_impact = pos_pred_rate.ratio.clone("Disparate Impact")
acceptance_rate = precision.clone("Acceptance Rate")
cond_accept = GroupMetric(
    lambda TP, FP, FN, TN: (TP + FN) / ge1(TP + FP), "Conditional Acceptance Rate"
)
cond_reject = GroupMetric(
    lambda TP, FP, FN, TN: (TN + FP) / ge1(TN + FN), "Conditional Rejectance Rate"
)
specificity = true_neg_rate.clone("Specificity")
rejection_rate = neg_pred_val.clone("Rejection Rate")
error_ratio = GroupMetric(lambda TP, FP, FN, TN: FP / ge1(FN), "Error Ratio")
treatment_equality = error_ratio.diff.clone("Treatment Equality")

gen_entropy = GroupMetric(
    lambda TP, FP, FN, TN: (
        (TP + FP + TN + FN) * (TP + FP * 4 + TN) / ge1(TP + FP * 2 + TN) ** 2 - 1
    )
    / 2,
    "Generalized Entropy",
    False,
)
clarify_metrics = {
    # "class_imbalance": class_imbalance,
    "demographic_parity": demographic_parity,
    "disparate_impact": disparate_impact,
    "cond_accept.diff": cond_accept.diff,
    "cond_reject.diff": cond_reject.diff,
    "accuracy.diff": accuracy.diff,
    "recall.diff": recall.diff,
    "acceptance_rate.diff": acceptance_rate.diff,
    "specificity.diff": specificity.diff,
    "rejection_rate.diff": rejection_rate.diff,
    "treatment_equality": treatment_equality,
    # gen_entropy": gen_entropy,
}


# Existing fairness definitions.
# Binary definitions from: https://fairware.cs.umass.edu/papers/Verma.pdf
# As all definitions just say 'these should be equal' we report the max difference in values
# as a measure of inequality.

statistical_parity = demographic_parity.clone("Statistical Parity")
predictive_parity = precision.diff.clone("Predictive Parity")
predictive_equality = false_neg_rate.diff.clone("Predictive Equality")
equal_opportunity = recall.diff.clone("Equal Opportunity")
equalized_odds = AddGroupMetrics(
    true_pos_rate.diff, true_neg_rate.diff, "Equalized Odds"
)
equalized_odds_max = MaxGroupMetrics(
    true_pos_rate.max_diff, true_neg_rate.max_diff, "Equalized Odds (L_inf)"
)

cond_use_accuracy = AddGroupMetrics(
    pos_pred_val.diff, neg_pred_val.diff, "Conditional Use Accuracy"
)
accuracy_parity = accuracy.diff.clone("Accuracy Parity")

verma_metrics = {
    "statistical_parity": statistical_parity,
    "predictive_parity": predictive_parity,
    "recall.diff": equal_opportunity,
    "miss_rate.diff": false_neg_rate.diff,
    "equalized_odds": equalized_odds,
    "cond_use_accuracy": cond_use_accuracy,
    # "predictive_equality": predictive_equality,
    "accuracy.diff": accuracy.diff,
    "treatment_equality": treatment_equality,
}

rate_metrics = {
    "pos_pred_rate": pos_pred_rate.diff,
    **{k: v.diff for k, v in standard_metrics.items()},
}

default_fairness_measures = verma_metrics

# Complex fairness definitions
# Here we define standard fairness measures that require more than per group difference/ratios

# We start with the unconditional forms of fairness from "why fairness can not be automated"
pos_pred_proportion = GroupMetric(lambda TP, FP, FN, TN, TTP, TFP, TFN, TTN: (TP+FP)/ge1(TTP+TFP)[:, np.newaxis],
                                  'Proportion of Positive Preditions', total_metric=True)
neg_pred_proportion = GroupMetric(lambda TP, FP, FN, TN, TTP, TFP, TFN, TTN: (TN+FN)/ge1(TTN+TFN)[:, np.newaxis],
                                  'Proportion of Negative Predictions', total_metric=True)
diff_pred_proportion = GroupMetric(lambda TP, FP, FN, TN, TTP, TFP, TFN, TTN: ((TP+FP)/ge1(TTP+TFP)[:, np.newaxis])
                                   - (TN+FN)/ge1(TTN+TFN)[:, np.newaxis],
                                   'Difference of Proportions of Predictions', total_metric=True)

abs_diff_pred_proportion = GroupMetric(lambda TP, FP, FN, TN, TTP, TFP, TFN, TTN: np.abs((TN+FN)/ge1(TTN+TFN)[:, np.newaxis]
                                                                                         - (TP+FP)/ge1(TTP+TFP)[:, np.newaxis]),
                                       'Proportion of Negative Predictions', total_metric=True)


pos_data_proportion = GroupMetric(lambda TP, FP, FN, TN, TTP, TFP, TFN, TTN: (TP+FN)/ge1(TTP+TFN)[:, np.newaxis],
                                  'Proportion of Positive Labels', total_metric=True)
neg_data_proportion = GroupMetric(lambda TP, FP, FN, TN, TTP, TFP, TFN, TTN: (TN+FP)/ge1(TTN+TFP)[:, np.newaxis],
                                  'Proportion of Negative Labels', total_metric=True)
diff_data_proportion = GroupMetric(lambda TP, FP, FN, TN, TTP, TFP, TFN, TTN: (TP+FN)/ge1(TTP+TFN)[:, np.newaxis]-(TN+FP)/ge1(TTN+TFP)[:, np.newaxis],
                                   'Difference of Proportion of Labels', total_metric=True)

abs_diff_data_proportion = GroupMetric(lambda TP, FP, FN, TN, TTP, TFP, TFN, TTN: np.abs((TP+FN)/ge1(TTP+TFN)[:, np.newaxis]
                                                                                         - (TN+FP)/ge1(TTN+TFP)[:, np.newaxis]),
                                       'Absolute Difference of Proportion of Labels', total_metric=True)

wachter_measures = {'pos_data_proportion': pos_data_proportion,
                    'neg_data_proportion': neg_data_proportion,
                    'diff_data_proportion': diff_data_proportion,
                    'pos_pred_proportion': pos_pred_proportion,
                    'neg_pred_proportion': neg_pred_proportion,
                    'diff_pred_proportion': diff_pred_proportion}

# directed bias amplification
# see: https://proceedings.mlr.press/v139/wang21t/wang21t.pdf
# warning do not use when enforcing fairness


def _y_a_t(TP, FP, FN, TN, TTP, TFN, TFP, TTN):
    "internal helper function that computes y_a_t from Wang et al"
    # p(A_a=1,T=1)
    # we remove one common normalisation term from PAT and PA as they cancel anyway
    PAT = (TP+FN)  # /(TTP+TFN+TFP+TTN)
    # p(A_a=1)p(T=1)
    PA = (TP+FP+FN+TN)  # /(TTP+TFN+TFP+TTN)
    PT = (TTP+TFN)/(TTP+TFN+TFP+TTN)
    out = 2*(PAT > PA*PT[:, np.newaxis]) - 1
    return out


directed_bias_amplification = GroupMetric(lambda TP, FP, FN, TN, TTP, TFN, TFP, TTN: _y_a_t(TP, FP, FN, TN, TTP, TFN, TFP, TTN) *
                                          (FN - FP)/(TP + FP + FN + TN),
                                          'Directed Bias Amplification', total_metric=True, greater_is_better=False)
