# OxonFair: An Algorithmic Fairness Toolkit for High-Capacity Models

OxonFair is an expressive toolkit designed to enforce a wide-range of fairness definitions and to customize binary classifier behavior.
The toolkit is designed to overcome a range of shortcomings in existing fairness toolkits for high-capacity models that overfit to the training data.
Unlike other toolkits it is designed and works for computer vision and NLP problems alongside tabular data.

Check out the colab demo [here](https://colab.research.google.com/drive/1CfcS3AX7M2MO1wW33wU1LDiY5DwtyyxH?usp=sharing) or read the [preprint](https://arxiv.org/abs/2407.13710).

For low-capacity models (e.g., logistic regression over a small number of variables, and decision-trees of limited depth), we recommend [fairlearn](https://github.com/fairlearn/fairlearn).

We support a range of complex classifiers including [pytorch](https://pytorch.org/), [scikit learn](https://scikit-learn.org/stable/), and ensembles provided by [autogluon](https://auto.gluon.ai/stable/index.html).

OxonFair is a continuation of early work done at [autogluon.fair](https://github.com/autogluon/autogluon-fair) and actively maintained.

## Source install

### Standard install

In the terminal type:

    pip install 'oxonfair[full]'

This will download and install enough code to run any notebooks except those comparing with fairlearn. This includes autogluon, pytorch, and XGBoost. If this is too many dependencies, try a minimal install.

### Minimal install

In the terminal type:

    pip install oxonfair

By default, this will only install the necessary dependencies sklearn; pandas; and numpy. You will have to install `matplotlib` to plot.

### Compare with Fairlearn

In the terminal type:

    pip install 'oxonfair[notebooks]'

This will download enough supporting libraries, including fairlearn, to run all the notebooks.

### Full install for running the test suite

Download the source of OxonFair and in the source directory run:

    pip install -e .\[tests\]

You probably don't want to install this unless you're looking to modify the codebase.

## Examples

Now run the [Example Notebook](./examples/quickstart_autogluon.ipynb) or try some of the example below.

For scikit/XGBoost, see [sklearn.md](./sklearn.md) and the [Example Notebook](./examples/quickstart_xgboost.ipynb)

For pytorch, see a toy example on [adult](./examples/pytorch_minimal_demo.ipynb) and for computer vision, this [Example Notebook](./examples/quickstart_DeepFairPredictor_computer_vision.ipynb)

More demo notebooks are present in the [examples folder](./examples/README.md).

## Demo using XGBoost

    # Load libraries
    from oxonfair import dataset_loader, FairPredictor
    from oxonfair import group_metrics as gm
    import xgboost

    # Download and partition the adult dataset into training and test data
    train_data, _, test_data = dataset_loader.adult('sex', train_proportion=0.7, test_proportion=0.3)
    # Train an XGBoost classifier on the training set                                              
    predictor = xgboost.XGBClassifier().fit(X=train_data['data'], y=train_data['target'])

    # Specify that we want to create a fair predictor by reusing the training set
    # (at the risk of overfitting) to enforce fairness with respect to sex.

    fpredict = FairPredictor(predictor, train_data, 'sex')

    # Enforce demographic parity to within 2%
    fpredict.fit(gm.accuracy, gm.demographic_parity, 0.02)
    
    # Evaluate per_group on the test set
    fpredict.evaluate_groups(test_data)

## Overview

OxonFair is a postprocessing approach for enforcing fairness, with support for a wide range of performance metrics and fairness criteria, and support for inferred attributes, i.e., it does not require access to protected attributes at test time.
Under the hood, FairPredictor works by adjusting the decision boundary for each group individually. Where groups are not available, it makes use of inferred group membership to adjust decision boundaries.

The key idea underlying this toolkit is that for a wide range of use cases, the most suitable classifier should do more than maximize some form of accuracy.
We offer a general toolkit that allows different measures to be optimized and additional constraints to be imposed by tuning the behavior of a binary predictor on validation data.

For example, classifiers can be tuned to maximize performance for a wide range of metrics such as:

* Accuracy
* Balanced Accuracy
* F1 score
* MCC
* Custom utility functions

While also approximately satisfying a wide range of group constraints such as:

* Demographic Parity (The idea that positive decisions should occur at the same rates for all protected groups, for example for men at the same rate as for women)
* Equal Opportunity (The recall should be the same for all protected groups)
* Minimum Recall Constraints (The recall should be above a particular level for all groups)
* Minimum Precision Constraints (The precision should be above a particular level for all groups)
* Custom Fairness Metrics

The full set of constraints and objectives can be seen in the list of measures in [measures.md](./measures.md).

### Why Another Fairness Library?

Fundamentally, most existing fairness methods are not appropriate for use with complex classifiers on high-dimensional data. These classifiers are prone to overfitting on the training data, which means that trying to balance error rates (e.g., when using equal opportunity) on the training data, is unlikely to transfer well to new unseen data. This is a particular problem when using computer vision (see [Zietlow et al.](https://arxiv.org/abs/2203.04913)), but can also occur with tabular data. Moreover, iteratively retraining complex models (a common requirement of many methods for enforcing fairness) is punitively slow when training the model once might take days, or even weeks, if you are trying to maximize performance.

At the same time, postprocessing methods which allow you to train once, and then improve fairness on held-out validation data generally require the protected attributes to be available at test time, which is often infeasible, particularly with computer vision.

OxonFair is built from the ground up to avoid these issues. It is a postprocessing approach, explicitly designed to use inferred attributes where protected attributes are not available to enforce fairness. Fairness can be enforced both on validation, or on the train set, when you are short of data and overfitting is not a concern. When enforcing fairness in deep networks or using provided attributes, a classifier is only trained once, for non network-based approaches, e.g., scikit-learn or xgboost, with inferred attributes we require the training of two classifier (one to predict the original task, and a second to estimate groups membership).

That said, we make several additional design decisions which we believe make for a better experience for data scientists:

#### Direct support for pytorch including NLP and Computer Vision

See [here](./examples/pytorch_minimal_demo.ipynb) for an example.
In brief, the steps are:

1. Train your network.
2. Call `DeepFairPredictor` with the network output on validation data.
3. Call `fit` to enforce fairness. Use `evaluate_groups` and `plot_frontier` to explore trade-offs.
4. Use `merge_heads_pytorch` to generate a fair network.  

#### Fine-grained control of behavior

##### Wide Choice of performance measure

Unlike other approaches to fairness, FairPredictor allows the optimization of arbitrary performance measures such as F1 or MCC, subject to fairness constraints. This can substantially improve the fairness/performance trade-off with, for example, F1 scores frequently being 3-4% higher when directly optimized for rather than accuracy.

##### Wide Choice of Fairness Measures

Rather than offering a range of different fairness methods that enforce a small number of fairness definitions through a variety of different methods, we offer one method that can enforce a much wider range of fairness definitions out of the box, alongside support for custom fairness definitions.

Of the set of decision-based group-metrics  discussed in [Verma and Rubin](https://fairware.cs.umass.edu/papers/Verma.pdf), and the metrics measured by [Sagemaker Clarify](https://pages.awscloud.com/rs/112-TZM-766/images/Fairness.Measures.for.Machine.Learning.in.Finance.pdf), out of the box FairPredictor offers the ability to both measure and enforce all of the 8 group metrics used to evaluate classifier decision measured in Verma and Rubin, and all 12 group measures used to evaluate decisions in Clarify.

##### Direct Remedy of Harms

See this [paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4331652) for the problems this addresses.

An Ipython notebook generating many of the figures in the paper can be found here: [Levelling up notebook](examples/levelling_up.ipynb)

Many fairness measures can be understood as identifying a harm, and then equalizing this harm across the population as a whole. For example, the use of demographic parity of identifies people as being harmed by a low selection rate, which is then set to be the same for all groups. Similarly, equal opportunity identifies people as being harmed by low recall, and balances this harm across all groups. However, these fairness formulations often implicitly *level down* and achieve fairness, by decreasing recall for some of the groups.

As an alternative to equalizing the harm across the population, we allow data scientists to specify minimum rates of e.g., recall, precision, or selection rate for every group, with one line of code. E.g.

    fpredictor.fit(gm.accuracy, gm.precision.min, 0.5)

 finds a classifier that maximizes accuracy while having a precision of at least 0.5 for every group.

These constraints have wider uses outside of fairness. For example, a classifier can be trained to identify common defects across multiple factory lines, and its behavior can be altered to enforce a recall of 90% of all defects for each line (at a cost of increased false positives).

    fpredictor.fit(gm.accuracy, gm.recall.min, 0.9)

##### Support for Utility based approaches

We provide support for the utility based approach set out in [Fairness On The Ground: Applying Algorithmic Fairness Approaches To Production Systems](https://arxiv.org/pdf/2103.06172.pdf), whereby different thresholds can be selected per group to optimize a utility-based objective.

Utility functions can be defined in one line.

For example, consider a situation where an ML system identifies potential problems that require intervening. 

Every intervention may have a cost of 1, regardless of if it was needed, but a missed intervention that was needed has a cost of 5. Finally, not making an intervention when one was unneeded has a cost of 0. This can be written as:

    my_utility=gm.Utility([1, 1, 5, 0], 'Testing Costs')

and optimized alongside fairness or performance constraints. For example,

    fpredictor.fit(my_utility)

optimizes the utility, while

    fpredictor.fit(my_utility, gm.accuracy, 0.5)

optimizes the utility subject to the requirement that the classifier accuracy can not drop below 0.5.

##### Support for user-specified performance and fairness measures

As well as providing support for enforcing a wide range of performance and fairness measures, we allow users to define their own metrics and fairness measures.

For example, a custom implementation of recall can be defined as:

    my_recall = gm.GroupMetric(lambda TP, FP, FN, TN: (TP) / (TP + FN), 'Recall')

and then the average difference in recall between groups (corresponding to the fairness definition of Equal Opportunity) is provided by calling `my_recall.diff`, and the minimum recall over any group (which can be used to ensure that the recall is above a particular value for every group) is given by `my_recall.min`.

### Altering Behavior

The behavior of classifiers can be altered through the fit function.
Given a pretrained binary predictor, we define a fair classifier that will allow us to alter the existing behavior on a validation dataset, by using a labeled attribute 'sex'.

    fpredictor = FairPredictor(predictor,  validation_data, 'sex')

the fit function takes three arguments that describe an objective such as accuracy or F1 that should be optimized, and a constraint such as the demographic parity violation should be below 2%.

This takes the form:

    fpredictor.fit(gm.f1, gm.demographic_parity, 0.02)

The constraint and objective can be swapped so the code:

    fpredictor.fit(gm.demographic_parity, gm.f1, 0.75)

will find the method with the lowest demographic parity violation such that F1 is above 0.75.
If, for example, you wish to optimize F1 without any additional constraints, you can just call:

    fpredictor.fit(gm.f1)

The default behavior (we should minimize the demographic parity violation, but maximize F1) is inferred from standard usage but can be overwritten by setting the optional parameters `greater_is_better_obj` and `greater_is_better_const` to `True` or `False`.

Where constraints cannot be satisfied, for example, if we require that the F1 score must be above 1.1, `fit` returns the solution closest to satisfying it.

For more examples using fit see [using_fit.md](./using_fit.md).

### Measuring Behavior

A key challenge in deciding how to alter the behavior of classifiers is that these decisions have knock-on effects elsewhere. For example, increasing the precision of classifiers, will often decrease their recall and vice versa.

In the same way, many fairness definitions may be at odds with one another, and increasing the fairness with respect to one definition can decrease it with respect to other definitions.

As such, we offer a range of methods for evaluating the performance of classifiers.

    fpredictor.evaluate(data (optional), groups (optional), dictionary_of_methods (optional), verbose=False)

By default, this method reports the standard binary evaluation criteria of autogluon for both the original and updated predictor, over the data used by fit. The behavior can be altered by providing either alternate data or a new dictionary of methods. Where groups are not provided, it will use the same groups as passed to `fit`, but this can be altered. If verbose is set to true, the table contains the long names of methods, otherwise it reports the dictionary keys.

    fpredictor.evaluate_fairness(data (optional), groups (optional), dictionary_of_methods (optional), verbose=False)

By default, this method reports the standard fairness metrics of SageMaker Clarify for both the original and updated predictor, over the data used by fit. The behavior can be altered by providing either alternate data or a new dictionary of methods. Where groups is not provided, it will use the same groups as passed to `fit`, but this can be altered. If verbose is set to true, the table contains the long names of methods, otherwise it reports the dictionary keys.

    fpredictor.evaluate_groups(data (optional), groups (optional), dictionary_of_methods (optional), return_original=False, verbose=False)

By default this method reports, per group, the standard binary evaluation criteria of autogluon for both the updated predictor only, over the data used by fit. The behavior can be altered by providing either alternate data or a new dictionary of methods. Where groups is not provided, it will use the same groups as passed to `fit`, but this can be altered. If you wish to also see the per group performance of the original classifier, use `return_original=True` to receive a dict containing the per_group performance of the original and updated classifier. If verbose is set to true, the table contains the long names of methods, otherwise it reports the dictionary keys.

### Fairness using Inferred Attributes

In many cases, the attribute you wish to be fair with respect to such as `sex` may not be available at test time. In this case you can make use of inferred attributes predicted by another classifier. This can be done by defining the fair predictor in the following way.

    fpredictor = fair.FairPredictor(predictor,  data, 'sex', inferred_groups=attribute_pred)

where `attribute_pred` is another autogluon predictor trained to predict the attribute, such as `sex` you wish to infer.
Then `fpredictor` can be used in same way described above.

Note that the labeled attribute is used to evaluate fairness, and the use of the inferred attributes are tuned to optimize fairness with respect to the labeled attributes. This means that even if the inferred attributes are not that accurate, they can be still used to enforce fairness, albeit with a drop in performance.

To make it easier to use inferred attributes, we provide a helper function for autogluon:

    predictor, attribute_pred = fair.inferred_attribute_builder(train_data, 'class', 'sex')

This allows for the easy training of two tabular classifiers, one called `predictor` to predict the target label `class` without using the attribute `sex` and one called `attribute_pred` to predict `sex` without using the the target label.

### Fairness on COMPAS using Inferred Attributes

We demonstrate how to enforce a wide range of fairness definitions on the COMPAS dataset. This dataset records paroles caught violating the terms of parole. As it measures who was caught, it is strongly influenced by policing and environmental biases, and should not be confused with a measurement of who actually violated their terms of parole. See [this paper](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/92cc227532d17e56e07902b254dfad10-Paper-round1.pdf) for a discussion of its limitations and caveats.

See this [notebook](./examples/compas_autogluon.ipynb) for details.

### Best Practices

It is common for machine learning algorithms to overfit training data. Therefore, if you want your fairness constraints to carry over to unseen data we recommend that they are enforced on a large validation set, rather than the training set. For low-dimensional datasets, many classifiers, with a careful choice of hyperparameter, are robust to overfitting and fairness constraints enforced on training data can carry over to unseen test data. In fact, given the choice between enforcing fairness constraints on a large training set, vs. using a significantly smaller validation set, reusing the training set may result in better generalization of the desired behavior to unseen data. However, this behavior is not guaranteed, and should always be empirically validated.

#### Challenges with unbalanced data

Many datasets are unbalanced both in the size of protected groups and in the prevalence of positive or negatively labeled data. When a rare group rarely receives positives outcomes, large datasets are needed to correctly estimate the rate of failure per group on positive data. This can make it very hard to reliably enforce or evaluate measures such as equal opportunity or minimum recall on unbalanced datasets, particularly where the baseline classifier has relatively high accuracy. The size and nature of the dataset needs to be carefully considered when choosing a fairness metric.

For example, on the historic dataset 'adult'; African Americans, despite being the second largest ethnicity after Caucasian, only make up around 10% of the dataset, with only 10% of them earning over 50k (N.B. adult is based on census data from 1994, in the same dataset, around 20% of white people earned over 50k). If an algorithm had a 10% error rate on this subset of the data, we are concerned with the behavior of around 10%^3 i.e., 0.1% of the data. This problem becomes even greater when looking at less prevalent groups.

For this reason, reliably guaranteeing high-accuracy across all groups, or that fairness measures are satisfied, requires access to rebalanced datasets, or much larger datasets than are needed for guaranteeing accuracy at the population level.

[quickstart_autogluon.ipynb](./examples/quickstart_autogluon.ipynb) has an example on the adult dataset where demographic parity is only weakly enforced on test data for the smaller groups `American-Indian-Eskimo`, and `Asian-Pacific-Islander` due to limited sample size.

### List of Measures

See [measures.md](./measures.md) for a full list of fairness and performance measures supported by OxonFair.
