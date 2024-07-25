"""Definitions of conditional measures for fairness and performance"""

from .group_metric_classes import (
    ConditionalWeighting,
)  # pylint: disable=unused-import # noqa
from . import group_metrics as gm


def reweight_by_factor_size(group_positives, group_negatives,
                            intersectional_positives, intersectional_negatives):
    """Used to rescore notions that depend on total number of entries, e.g. Positive Decision Rate"""
    return ((group_positives + group_negatives) /
            max(1, intersectional_positives + intersectional_negatives))


def reweight_by_factor_postives(group_positives, group_negatives,
                                intersectional_positives, intersectional_negatives):
    """Used to rescore notions that depend on total number of positive entries e.g. recall"""
    return (intersectional_positives) / max(1, group_positives)


def reweight_by_factor_negatives(group_positives, group_negatives,
                                 intersectional_positives, intersectional_negatives):
    """Used to rescore notions that depend on total number of negative entries e.g. sensitivity"""
    return (intersectional_negatives) / max(1, group_negatives)


def reweight_by_1(group_positives, group_negatives,
                  intersectional_positives, intersectional_negatives):
    """Used for debugging"""
    return 1


total_weights = ConditionalWeighting(reweight_by_factor_size)
pos_weights = ConditionalWeighting(reweight_by_factor_postives)
neg_weights = ConditionalWeighting(reweight_by_factor_negatives)
constant = ConditionalWeighting(reweight_by_1)


def build_cond_form(metric, weighting):
    "Build a conditional form from an existing metric using a given weighting"
    name = "Conditional " + metric.name
    return metric.clone(name, weighting)


pos_pred_rate = build_cond_form(gm.pos_pred_rate, total_weights)
neg_pred_rate = build_cond_form(gm.neg_pred_rate, total_weights)
pos_data_rate = build_cond_form(gm.pos_data_rate, total_weights)
neg_data_rate = build_cond_form(gm.neg_data_rate, total_weights)

true_pos_rate = build_cond_form(gm.true_pos_rate, pos_weights)
true_neg_rate = build_cond_form(gm.true_neg_rate, neg_weights)
false_pos_rate = build_cond_form(gm.false_pos_rate, neg_weights)
false_neg_rate = build_cond_form(gm.false_neg_rate, neg_weights)


accuracy = build_cond_form(gm.accuracy, total_weights)

cond_measures = {
    "accuracy": accuracy,
    "true_pos_rate": true_pos_rate,
    "true_neg_rate": true_neg_rate,
    "false_pos_rate": false_pos_rate,
    "false_neg_rate": false_neg_rate,
    "pos_pred_rate": pos_pred_rate,
    "neg_pred_rate": neg_pred_rate,
    "pos_data_rate": pos_data_rate,
    "neg_data_rate": neg_data_rate,
}


cond_disparities = dict(
    (key + "_diff", item.diff) for (key, item) in cond_measures.items()
)

pos_data_proportion = build_cond_form(gm.pos_data_proportion, total_weights)
neg_data_proportion = build_cond_form(gm.neg_data_proportion, total_weights)
diff_data_proportion = build_cond_form(gm.diff_data_proportion, total_weights)
abs_diff_data_proportion = build_cond_form(gm.abs_diff_data_proportion, total_weights)
pos_pred_proportion = build_cond_form(gm.pos_pred_proportion, total_weights)
neg_pred_proportion = build_cond_form(gm.neg_pred_proportion, total_weights)
diff_pred_proportion = build_cond_form(gm.diff_pred_proportion, total_weights)
abs_diff_pred_proportion = build_cond_form(gm.abs_diff_pred_proportion, total_weights)

wachter_measures = {'pos_data_proportion': pos_data_proportion,
                          'neg_data_proportion': neg_data_proportion,
                          'diff_data_proportion': diff_data_proportion,
                          'pos_pred_proportion': pos_pred_proportion,
                          'neg_pred_proportion': neg_pred_proportion,
                          'diff_pred_proportion': diff_pred_proportion}
