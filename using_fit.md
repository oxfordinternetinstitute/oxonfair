# Using Fit in OxonFair

OxonFair is a more flexible toolkit than other fairness approaches, and this expressiveness takes some getting used to. To show how it works, we are going to give examples using fit to enforce a range of different fairness definitions. For these examples, we're going to maximise accuracy, but if that doesn't meet your use case feel free to switch it another performance measure like balanced accuracy (`gm.balanced_accuracy`), or F1 (`gm.f1`).

The use of `fit` is generic. The calls we show here work on tabular, image, and NLP data either using inferred protected attributes or explicitly provided ones.

To get started we're just going to use explicit attributes on an XGBoost trained classifier on  tabular data. We won't worry about overfitting and will just enforce fairness on the training set.

Here is some sample code to train the base classifier on the adult dataset, and to prepare the fair classifier.

    from oxonfair import dataset_loader, FairPredictor
    from oxonfair import group_metrics as gm
    import xgboost
    train_data, _, test_data = dataset_loader.adult('sex',train_ratio=0.7,
                                                  test_ratio=0.3)
    predictor = xgboost.XGBClassifier().fit(X=train_data['data'],
                                           y=train_data['target'])
    fpredict = FairPredictor(predictor, train_data, 'sex')
To see the trade-offs made by OxonFair, after running fit, you can call:

    fpredict.evaluate_groups(test_data)
This will show how the classifier behaviour is altered on a group-by-group basis. Calling

    fpredict.plot_frontier()
Will show the Pareto frontier, and how you can expect the constraint and objective to vary as you alter the number. `fpredict.plot_frontier(test_data)` will reevaluate the frontier on test data and show how much you are overfitting to noise.

Now let's look at some example uses for `.fit`.

* Enforce Demographic Parity to within 2%:
      `fpredict.fit(gm.accuracy, gm.demographic_parity, 0.02)`
    or because demographic parity just says the decision rate should be the same for each group, we can write that as the difference between groups being less than 2%:
      `fpredict.fit(gm.accuracy, gm.positive_decision_rate.diff, 0.02)`
* Enforce that the Disparate Impact ratio is over 80%:
      `fpredict.fit(gm.accuracy, gm.disparate_impact, 0.80)`
    because disparate impact is just the ratio of positive decisions, we can also write it as:
      `fpredict.fit(gm.accuracy, gm.positive_decision_rate.ratio, 0.80)`
    Note that fit alters its behaviour depending on what you pass it. By default performance measures like `accuracy`  are maximised; differences are minimised; and ratios are maximised.  If you don't like this behaviour you can override it by setting  `obj_greater_is_better` or `const_greater_is_better` to True or False.
* Enforce Equal Opportunity to within 1%:
      `fpredict.fit(gm.accuracy, gm.equal_opportunity, 0.01)`
    as equal opportunity is just defined as the difference in recall, this is the same as:
      `fpredict.fit(gm.accuracy, gm.recall.min, 0.01)`
* Enforce recall ratio is within 80%
      `fpredict.fit(gm.accuracy, gm.recall.ratio, 0.80)`
    This definition of fairness doesn't even have a name, but using a ratio instead of difference is useful for problems where the selection rate gets very small.
* Enforce Equalized Odds to within 2%:
      `fpredict.fit(gm.accuracy, gm.equalized_odds, 0.02)`
    Under the hood we define equalized odds as the average of the group difference in recall (i.e. True Positive Rates) and the group difference in True Negative Rates
* Enforce that the difference in precision is less than 5%
        `fpredict.fit(gm.accuracy, gm.precision.diff, 0.05)`
    We could also look this up in [Verma and Rudin](ww.) and find out that this has the name predictive parity so, this code will also work.
        `fpredict.fit(gm.accuracy, gm.predictive_parity, 0.05)`
* Enforcing that the recall rate is  at least 80% for every group
          `fpredict.fit(gm.accuracy, gm.recall.min, 0.8)`
      This is useful because a key problem with fairness is that it tends to [level-down](https://arxiv.org/pdf/2302.02404). When you enforce equal opportunity it will typically improve recall rates for disadvantaged groups, but for the groups with high recall, recall and accuracy will also drop. To avoid this, we can simply"level-up" and push-up the recall for every disadvantaged group, while leaving the classifier alone where it already works acceptably well.
* Enforce that the selection rate is over 40% for every group.
          `fpredict.fit(gm.accuracy, gm.positive_decision_rate, 0.4)`
    This is the levelling-up version of demographic parity.
* Pareto efficient minimax fairness
    This enforces that the accuracy over the worst performing  pair of (target label, group) is as high as possible, while maximising the overall accuracy (see [Martenez et al.](https://arxiv.org/abs/2011.01821))
          `fpredict.fit(gm.min_accuracy.min, gm.accuracy, 0)`
    You can also simply maximise the accuracy for the worst performing group while also maximising global accuracy using:
        `fpredict.fit(gm.accuracy.min, gm.accuracy, 0)`
    But for high-capacity models (see [Singh et al.](https://proceedings.mlr.press/v202/singh23b/singh23b.pdf) ) this is generally indistinguishable from just maximising accuracy.
* Enforce demographic parity, subject to the requirement that there is ~40% overall selection rate:
     `fpredict.fit(gm.positive_decision_rate.min, gm.positive_decision_rate, 0.4)`
    To understand why this works see the proof in [Goethal's et al.](https://arxiv.org/pdf/2406.01290).
* Enforce equal opportunity subject to the requirement that there is ~40% overall selection rate:
     `fpredict.fit(gm.recall.min, gm.positive_decision_rate, 0.4)`
* Enforce equal precision rates, subject to the requirement that there is ~40% overall selection rate:
      `fpredict.fit(gm.precision.min, gm.positive_decision_rate, 0.4, const_greater_is_better=True)`
    Here we must swap the sign on the constraint because precision is maximised as the selection rate goes to zero.
* Maximise utility:
      `utility = gm.utility(1, 1, 4, 0)`
      `fpredict.fit(utility)`
* Maximise utility while enforcing that the minimum group recall doesn't drop below 60%.
      `fpredict.fit(utility, gm.recall.min, 0.6)`
