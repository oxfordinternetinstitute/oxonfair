"""Definitions of conditional measures for fairness and performance"""
from .group_metric_classes import ConditionalWeighting # pylint: disable=unused-import # noqa
from .group_metrics import pos_pred_rate, recall, accuracy
import numpy as np
def reweight_by_factor_size(group_positives,group_negatives, intersectional_positives, intersectional_negatives):
    """Used to rescore notions that depend on total number of entries, e.g. Positive Decision Rate"""
    return (group_positives + group_negatives)/max(1,intersectional_positives + intersectional_negatives)


def reweight_by_factor_postives(group_positives,group_negatives, intersectional_positives, intersectional_negatives):
    """Used to rescore notions that depend on total number of positive entries e.g. recall"""
    return (intersectional_positives) / max(1, group_positives)


def reweight_by_factor_negatives(group_positives,group_negatives, intersectional_positives, intersectional_negatives):
    """Used to rescore notions that depend on total number of negative entries e.g. sensitivity"""
    return (intersectional_negatives) / max(1, group_negatives)

def reweight_by_5(group_positives,group_negatives, intersectional_positives, intersectional_negatives):
    """Used to rescore notions that depend on total number of negative entries e.g. sensitivity"""
    return 5



cond_total_weights =  ConditionalWeighting(reweight_by_factor_size)
cond_pos_weights = ConditionalWeighting(reweight_by_factor_postives)
cond_neg_weights = ConditionalWeighting(reweight_by_factor_negatives)
cond_debug = ConditionalWeighting(reweight_by_5)

#Todo: complete these
cond_pos_pred_rate = pos_pred_rate.clone('Conditional Positive Prediction Rate', cond_total_weights)
cond_accuracy = accuracy.clone('Conditional Accuracy', cond_total_weights)
cond_recall = recall.clone('Conditional Recall', cond_pos_weights)
