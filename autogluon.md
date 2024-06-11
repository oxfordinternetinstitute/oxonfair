# Using Autogluon and OxonFair

Here we show a simple example of enforcing fairness on an autogluon classifier.

The basic approach is the same as for [sklearn](./sklearn.md):

1. Fit a predictor.
2. Create a fairpredictor object using the predictor
3. call fit on the fairpredictor

    # Load data and train a baseline classifier
    from autogluon.tabular import TabularDataset, TabularPredictor
    from oxonfair import FairPredictor
    from oxonfair.utils import group_metrics as gm
    train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
    test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
    predictor = TabularPredictor(label='class').fit(train_data=train_data)

    # Modify predictor to enforce fairness over the train_data with respect to groups given by the column 'sex'
    fpredictor = FairPredictor(predictor,train_data,'sex')
    # Maximize accuracy while enforcing that the demographic parity (the difference in positive decision rates between men and women is at most 0.02)
    fpredictor.fit(gm.accuracy,gm.demographic_parity,0.02)

    # Evaluate on test data
    fpredictor.predict(test_data)

    # Evaluate a range of performance measures, and compare against original classifier on test data
    fpredictor.evaluate(test_data, verbose= True)

|                   |   original |   updated |
|:------------------|-----------:|----------:|
| Accuracy          |   0.876446 |  0.853926 |
| Balanced Accuracy |   0.796708 |  0.757129 |
| F1 score          |   0.712414 |  0.650502 |
| MCC               |   0.640503 |  0.568616 |
| Precision         |   0.795636 |  0.752408 |
| Recall            |   0.644953 |  0.572908 |
| roc_auc           |   0.931573 |  0.827535 |

    # Evaluate against a range of standard fairness definitions and compare against original classifier on test data
    fpredictor.evaluate_fairness(test_data, verbose= True)

|                                                         |   original |    updated |
|:--------------------------------------------------------|-----------:|-----------:|
| Class Imbalance                                         |  0.195913  | 0.195913   |
| Demographic Parity                                      |  0.166669  | 0.00744171 |
| Disparate Impact                                        |  0.329182  | 0.959369   |
| Maximal Group Difference in Accuracy                    |  0.0936757 | 0.0684973  |
| Maximal Group Difference in Recall                      |  0.0590432 | 0.326703   |
| Maximal Group Difference in Conditional Acceptance Rate |  0.0917708 | 1.04471    |
| Maximal Group Difference in Acceptance Rate             |  0.0174675 | 0.347018   |
| Maximal Group Difference in Specificity                 |  0.0518869 | 0.0594707  |
| Maximal Group Difference in Conditional Rejectance Rate |  0.0450807 | 0.229982   |
| Maximal Group Difference in Rejection Rate              |  0.0922794 | 0.157476   |
| Treatment Equality                                      |  0.0653538 | 5.07559    |
| Generalized Entropy                                     |  0.0666204 | 0.080265   |

    # Evaluate a range of performance measures per group, and compare against original classifier on test data
    fpredictor.evaluate_groups(test_data, verbose= True, return_original=True)

|                                    |   Accuracy |   Balanced Accuracy |   F1 score |       MCC |   Precision |    Recall |   roc_auc |   Positive Count |   Negative Count |   Positive Label Rate |   Positive Prediction Rate |
|:-----------------------------------|-----------:|--------------------:|-----------:|----------:|------------:|----------:|----------:|-----------------:|-----------------:|----------------------:|---------------------------:|
| ('original', 'Overall')            |  0.876446  |          0.796708   | 0.712414   | 0.640503  |   0.795636  | 0.644953  | 0.931573  |             2318 |             7451 |              0.237281 |                 0.192343   |
| ('original', ' Female')            |  0.938583  |          0.787403   | 0.675241   | 0.649242  |   0.780669  | 0.594901  | 0.949251  |              353 |             2936 |              0.107327 |                 0.0817878  |
| ('original', ' Male')              |  0.844907  |          0.790981   | 0.718881   | 0.619052  |   0.798137  | 0.653944  | 0.91321   |             1965 |             4515 |              0.303241 |                 0.248457   |
| ('original', 'Maximum difference') |  0.0936757 |          0.00357813 | 0.04364    | 0.03019   |   0.0174675 | 0.0590432 | 0.0360405 |             1612 |             1579 |              0.195913 |                 0.166669   |
| ('updated', 'Overall')             |  0.853926  |          0.757129   | 0.650502   | 0.568616  |   0.752408  | 0.572908  | 0.827535  |             2318 |             7451 |              0.237281 |                 0.180674   |
| ('updated', ' Female')             |  0.899362  |          0.877586   | 0.644468   | 0.614161  |   0.519031  | 0.849858  | 0.949251  |              353 |             2936 |              0.107327 |                 0.175737   |
| ('updated', ' Male')               |  0.830864  |          0.74397    | 0.652284   | 0.579829  |   0.866049  | 0.523155  | 0.91321   |             1965 |             4515 |              0.303241 |                 0.183179   |
| ('updated', 'Maximum difference')  |  0.0684973 |          0.133616   | 0.00781595 | 0.0343327 |   0.347018  | 0.326703  | 0.0360405 |             1612 |             1579 |              0.195913 |                 0.00744171 |
