# Measures supported by OxonFair

OxonFair uses a wide range of measures to enforce and measures fairness and performance.

These measures can be passed to a `FairPredictor` by calling `FairPredictor.fit(objective, constraint, value)`.
This will optimise the measure `objective` subject to the requirement that the other measure `constraint` is greater or less than `value`, as required.

These measures can also evaluated by passing to the evaluation functions `evaluate`, `evaluate_groups`, and `evaluate_fairness` as a dict of measures, where the keys of the dict are short-form names using when `verbose=False` and the values are measures.

This document lists the standard measures provided by the group_metrics library, which is imported as:

    from oxonfair.utils import group_metrics as gm

## Basic Structure

The majority of measures are defined as GroupMetrics or sub-objects of GroupMetrics.

A group measure is specified by a function that takes the number of True Positives, False Positives, False Negatives, and True Negatives and returns a score; A string specifying the name of the of the measure; and optionally a bool indicating if greater values are better than smaller ones. For example, accuracy is defined as:

    accuracy = gm.GroupMetric(lambda TP, FP, FN, TN: (TP + TN) / (TP + FP + FN + TN), 'Accuracy')

For efficiency, our approach relies on broadcast semantics and all operations in the function must be applicable to numpy arrays.

Having defined a GroupMetric it can be called in two ways. Either:

    accuracy(target_labels, predictions, groups)

Here target_labels and predictions are binary vectors corresponding to either the target ground-truth values, or the predictions made by a classifier, with 1 representing the positive label and 0 otherwise. Groups is simply a vector of values where each unique value is assumed to correspond to a distinct group.

The other way it can be called is by passing it a single 3D array of dimension 4 by number of groups by k, where k is the number of candidate classifiers that the measure should be computed over.

As a convenience, GroupMetrics automatically implements a range of functionality as sub-objects.

Having defined a metric as above, we have a range of different objects:

* `metric.diff` reports the average absolute difference of the method between pairs of groups.
* `metric.average` reports the average of the method taken over all groups.
* `metric.max_diff` reports the maximum difference of the method between any pair of groups.
* `metric.max` reports the maximum value for any group.
* `metric.min` reports the minimum value for any group.
* `metric.overall` reports the overall value for all groups combined, and is the same as calling `metric` directly
* `metric.ratio` reports the average ratio over pairs of distinct groups, where smallest value is divided by the largest
* `metric.per_group` reports the value for every group.

These can be passed directly to fit, or to the evaluation functions we provide.

The vast majority of fairness metrics are implemented as a `.diff` of a standard performance measure, and by placing a `.min` after any measure such as `recall` or `precision` it is possible to add constraints that enforce that the precision or recall is above a particular value for every group.
gm.

## Dataset Measures

| Name             | Definition                                                       |
|------------------|------------------------------------------------------------------|
| `gm.count`          | Total number of points in a dataset or group                     |
| `gm.pos_data_count` | Total number of positively labeled points in a dataset or group |
| `gm.neg_data_count` | Total number of negatively labeled points in a dataset or group |
| `gm.pos_data_rate`  | Ratio of positively labeled points to size of the group         |
| `gm.neg_data_rate`  | Ratio of negatively labeled points to size of the group         |

## Standard Prediction Measures

| Name             | Definition                                                                                                     |
|------------------|----------------------------------------------------------------------------------------------------------------|
| `gm.pos_pred_rate`  | Positive Prediction Rate: Ratio of the number of positively predicted points to the size of the group          |
| `gm.neg_pred_rate`  | Negative Prediction Rate: Ratio of the number of negatively predicted points to the size of the group          |
| `gm.true_pos_rate`  | True Positive Rate: Ratio of true positives divided by total positive predictions                              |
| `gm.true_neg_rate`  | True Negative Rate: Ratio of true negatives divided by total negative predictions                              |
| `gm.false_pos_rate` | False Positive Rate: Ratio of False Positives divided by total negative prediction                             |
| `gm.false_neg_rate` | False Negative Rate: Ratio of False Negatives divided by total positive predictions                            |
| `gm.pos_pred_val`   | Positive Predicted Value': Ratio of True Positives divided by the total number of points with positive label   |
| `gm.neg_pred_val`   | Negative Predicted Value': Ratio of True Negatives divided by the total number of points with a negative label |

## Core Performance Measures

| Name                | Definition                                                                                                                                                                               |
|---------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `gm.accuracy`          | Proportion of points correctly identified                                                                                                                                                 |
| `gm.balanced_accuracy` | The average of the proportion of points with a positive label correctly identified and the proportion of points with a negative label correctly identified                              |
| `gm.min_accuracy`      | The minimum of the proportion of points with a positive label correctly identified and the proportion of points with a negative label correctly identified (common in min-max fairness) |
| `gm.f1`                | F1 Score. Defined as:  (2 * TP) / (2 * TP + FP + FN)                                                                                                                                     |
| `gm.precision`         | AKA Positive Prediction Rate                                                                                                                                                             |
| `gm.recall`            | AKA True Positive Prediction Rate                                                                                                                                                        |
| `gm.mcc`               | Matthews Correlation Coefficient. See https://en.wikipedia.org/wiki/Phi_coefficient                                                                                                      |

## Additional Performance Measures

| Name              | Definition                                                                        |
|-------------------|-----------------------------------------------------------------------------------|
| `gm.acceptance_rate` | AKA precision AKA Positive Prediction Rate                                        |
| `gm.cond_accept`     | Conditional Acceptance Rate. The ratio of positive predictions to positive labels |
| `gm.cond_reject`     | Conditional Rejectance Rate. The ratio of negative predictions to negative labels |
| `gm.specificity`     | AKA True Negative Rate                                                            |
| `gm.rejection_rate`  | AKA Negative Predicted Value                                                      |
| `gm.error_ratio`     | The ratio of False Positives to False Negatives                                    |

## Fairness Measures Supported

[Sagemaker Clarify](https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-measure-post-training-bias.html) Measures

| Name                   | Definition                                                                                                                                                                                                                                                                                   |
|------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `gm.class_imbalance`      | Average difference between groups in Positive Data Rate                                                                                                                                                                                                                                      |
| `gm.demographic_parity`   | AKA Statistical Parity.  Average difference between groups in Positive Prediction Rate                                                                                                                                                                                                       |
| `gm.disparate_impact`     | The smallest Positive Prediction Rate of any group divided by the largest                                                                                                                                                                                                                    |
| `gm.accuracy.diff`        | Average difference between groups in Accuracy                                                                                                                                                                                                                                                |
| `gm.recall.diff`          | AKA Equal Opportunity. Average difference between groups in Recall                                                                                                                                                                                                                           |
| `gm.cond_accept.diff`     | Average difference between groups in Conditional Acceptance Rate                                                                                                                                                                                                                             |
| `gm.acceptance_rate.diff` | Average difference between groups in Acceptance Rate                                                                                                                                                                                                                                         |
| `gm.specificity.diff`     | Average difference between groups in Specificity  (or True Negative Rate)                                                                                                                                                                                                                    |
| `gm.cond_reject.diff`     | Average difference between groups in Conditonal Rejectance Rate                                                                                                                                                                                                                              |
| `gm.rejection_rate.diff`  | Average difference between groups in Rejection Rate (or Negative Predicted Value)                                                                                                                                                                                                            |
| `gm.treatment_equality`   | Average difference between groups in Error Ratio                                                                                                                                                                                                                                             |
| `gm.gen_entropy`          | This is the expected square of a particular utility function divided by its expected value, minus 1 and then divided by 2. The function takes the form: `TP*1+FP*2+FN*1`, where TP, FP, NP, and TN are the true positives, false positives, false negatives and true negatives respectively. |

Measures from [Verma and Rubin](https://fairware.cs.umass.edu/papers/Verma.pdf).

All the measures in Verma and Rubin are defined as strict equalities for two groups. We relax them into a continuous measure that reports the Average difference over any pair of groups between the left and right sides of the equality.
These relaxations take value 0 only if the equalities are satisfied for all pairs of groups.

| Name                     | Definition                                                                                            |
|--------------------------|-------------------------------------------------------------------------------------------------------|
| `gm.statistical_parity`  | AKA Demographic Parity. Average difference between groups in Positive Prediction Rate                 |
| `gm.predictive_parity`   | AKA Rejection Rate Difference. Average difference between groups in Precision                         |
| `gm.false_pos_rate.diff` | AKA  Specificity Difference. Average difference between groups in False Positive rate.                |
| `gm.false_neg_rate.diff` | AKA Equal Opportunity or Recall difference. Average difference between groups in False Negative Rate |
| `gm.equalized_odds`      | The average of `true_pos_rate.diff` and  `false_neg_rate.diff`                                        |
| `gm.cond_use_accuracy`   | The average of `pos_pred_val.diff` and `neg_pred_val.diff`                                            |
| `gm.predictive_equality` | Average difference in False Negative Rate                                                             |
| `gm.accuracy._parity`    | Average difference in Accuracy                                                                        |
| `gm.treatment_equality`  | Average difference between groups in Error Ratio                                                      |

## Conditional Metrics

OxonFair also supports conditional metrics.
These are used to compensate for accetable biases present in the data.
For example, in one [famous case](https://pubmed.ncbi.nlm.nih.gov/17835295/), Berkley showed a strong gender bias in admissions despite the fact that each department had minimal admissions bias with respect to gender. The cause underlying this was that women were disproportionately applying to departments with higher rejection rates.

To measure this correct for this bias we the follow the method set out in the chapter 1 questions of: [Statistics by Freedman et al.](https://www.goodreads.com/book/show/147358.Statistics), which [Wachter et al.](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3547922) applied to algorithmic fairness.

This measure compensates for the fact that different selection rates across groups may be driven by an acceptable factor that is correlated with the protected attributes. For example, in the Berkley case, it is acceptable that different departments should have different admissions rates, but the choice of department is correlated with gender.

This is also measured by [Amazon Clarify](https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-data-bias-metric-cddl.html) and
[IBM360](https://aif360.readthedocs.io/en/latest/modules/generated/aif360.sklearn.metrics.conditional_demographic_disparity.html#aif360.sklearn.metrics.conditional_demographic_disparity)

However, no other fairness toolkit optimizes it.
All of these measures are subtly different, but weight data in the same way.
Freedman et al. considers the weighted proportion of people in a particular group recieving positive decisions vs. the total number of people in the group.  

Wachter et al. examines the weighted proportion of [members of a protected group]  within the set of all people recieving a positive decision; and the same weighted proportion of [members of the protected group] within the set of all people recieving a negative decision. If this proportion is larger for the positive set, than the negative set, the group is doing disproportionately well, and if it smaller, the group is doing disproportionately badly.

Clarify and IBM360 measures the difference of the two measures in Wachter et al.

All methods are broadly equivilent in the sense that the difference between every pair of groups using Freedman's measure is zero if and only if the difference between positives and negatives measures of Wachter et al., for every group is zero.

For simplicity, we implement Freedman's measure. This give nautural extensions to difference in conditional selection rate, corresponding to conditional demographic parity, and average ratio in conditional selection rate, corresponding to disparate impact. Moreover, the levelling-up measures such as minimal conditional selection rate will also work, which is not the case for the measure of Wachter et al.

We assign a weight $w_i$ to an individual $i$ belonging to a particular protected group, and conditioning factor e.g. school as:

$$
w_i = \frac{\#\text{individuals with the same conditioning factor}}{\#\text{individuals belonging to the same group and conditioning factor}}
$$

The conditional positive decision rate is then given by:
$$ \frac {\text{wTP+ wFP}{wTP +wFP +wFN +wTN}$$ where wTP, wFP, wFN, wTN are the weighted sum of True Positives, False Positives using the weights $w_i$.

This can be used for levelling up, by enforcing minimum conditional selection rates, and enforcing conditional demographic parity.

The use of conditional metrics is somewhat more involved, as it requires the specification of a conditioning factor, alongside groups.
Here is a quick example using a conditional minimimal selection rate of 0.3.

    import oxonfair
    import xgboost
    from oxonfair import group_metrics as gm
    from oxonfair import conditional_group_metrics as cgm
    train,val,test = oxonfair.dataset_loader.adult()
    classifier = xgboost.XGBClassifier().fit(y=train['target'], X=train['data'])
    fpred = oxonfair.FairPredictor(classifier, val, conditioning_factor='education-num')
    fpred.fit(gm.accuracy, cgm.pos_pred_rate.min,0.3)
    fpred.evaluate_groups(metrics=cgm.cond_measures)

We support conditioning on range of linear measures.

1. `cgm.accuaracy` conditional accuracy which is weighted in the same way;
2. `cgm.positive_decision_rate` conditional positive decision rate
3. `cgm.positive_data_rate` conditional positive data rate
4. `cgm.false_neg_rate` conditional false negative rate
5. `cgm.false_pos_rate` conditional false positive rate

For false negative and false positive rate, we normalise by the total number of negatively or positively labelled points rather than the total number of points.
