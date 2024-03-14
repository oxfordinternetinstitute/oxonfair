import fairlearn.reductions as red
import numpy as np
#Comparision of oxonfair with fairlearn for decision trees, random forest, and boosting.
fair_met = ['dp','eo'][1]
for i in range(3,4):
    algorithm = ['tree','forest','boost','boost_extended','SVM','logreg'][i]
    if algorithm == 'tree':
        import sklearn.tree
        classifier_type = sklearn.tree.DecisionTreeClassifier
        classifier_string = 'Decision Trees'
    elif algorithm == 'forest':
        import sklearn.ensemble
        classifier_type = sklearn.ensemble.RandomForestClassifier
        classifier_string = 'Random Forests'
    elif algorithm == 'boost':
        import xgboost as xgb
        classifier_type = xgb.XGBClassifier
        classifier_string ='XGBoost'
    elif algorithm == 'logreg':
        import sklearn.linear_model
        classifier_type = sklearn.linear_model.LogisticRegression
        classifier_string = 'Logistic Regression'

    if algorithm == 'boost_extended':
        import xgboost as xgb
        classifier_type = xgb.XGBClassifier
        def build_classifier():
            return classifier_type(max_depth=2,n_estimators=400)
        classifier_string = 'Boosting (max depth 20)'
    elif algorithm == 'SVM':
        import sklearn.svm
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        SVC = sklearn.svm.SVC
        gamma=0.001
        def build_classifier():
            return make_pipeline(StandardScaler(), SVC(gamma=gamma, probability=True))
        subsample = 10
        classifier_string = 'SVMs'
    else :
        def build_classifier():
            return classifier_type()

    classifier = build_classifier()

    from matplotlib import pyplot as plt


    import oxonfair
    import pandas as pd
    from oxonfair import FairPredictor
    from oxonfair.utils import group_metrics as gm
    train_data = pd.read_csv('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
    test_data = pd.read_csv('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')

    #Merge and shuffle the data
    total_data = pd.concat([train_data,test_data])
    y = total_data['class'] == ' <=50K'
    total_data = total_data.drop(columns='class')
    total_data=pd.get_dummies(total_data)

    train = total_data.sample(frac=0.5)
    val_test = total_data.drop(train.index).copy()
    train_y = y.iloc[train.index]
    val_test_y =y.drop(train_y.index)
    val = val_test.sample(frac=0.5)
    test = val_test.drop(val.index)
    val_y=y.iloc[val.index]
    test_y=val_test_y.drop(val.index)
    if algorithm == "SVM":
        classifier.fit(train[::subsample],train_y[::subsample])
    else:
        classifier.fit(train,train_y)

    val_dict={'data':val, 'target':val_y}
    test_dict={'data':test,'target':test_y}
    if algorithm == 'tree':
        sk_fpred=FairPredictor(classifier,val_dict,'sex_ Female',add_noise=0.05)
    else:
        sk_fpred=FairPredictor(classifier,val_dict,'sex_ Female')
    if fair_met == 'dp':
        met = oxonfair.group_metrics.demographic_parity
        fl_met = red.DemographicParity
    else:
        met = oxonfair.group_metrics.equal_opportunity
        fl_met = red.TruePositiveRateParity
    sk_fpred.fit(oxonfair.group_metrics.accuracy,met,0.01)

    plt.figure()
    sk_fpred.plot_frontier(val_dict,new_plot=False,prefix="Val: OxonFair ")
    sk_fpred.plot_frontier(test_dict,new_plot=False,prefix="Test: OxonFair ")

    for eps in (0.01,0.02,0.05,0.10):
        classifier2 = build_classifier()
        if algorithm == "SVM":
            reduction =red.ExponentiatedGradient(classifier2, fl_met() ,sample_weight_name="svc__sample_weight", eps=eps)

            reduction.fit(train[::subsample], np.asarray(train_y)[::subsample], sensitive_features=np.asarray(train['sex_ Female'])[::subsample])
        else:
            reduction =red.ExponentiatedGradient(classifier2, fl_met(), eps=eps)
            reduction.fit(train, np.asarray(train_y), sensitive_features=np.asarray(train['sex_ Female']))

        preds=reduction.predict(test)
        preds_train=reduction.predict(train)
        preds_val=reduction.predict(val)

        if algorithm == "SVM":
            plt.scatter(met(np.asarray(train_y[::subsample]),preds_train[::subsample],np.asarray(train['sex_ Female'][::subsample])),
                    oxonfair.group_metrics.accuracy(np.asarray(train_y[::subsample]), preds_train[::subsample],np.asarray(train['sex_ Female'])[::subsample]),
                    label='Train: FairLearn eps='+str(eps), s=50,marker='p')
        else:
            plt.scatter(met(np.asarray(train_y),preds_train,np.asarray(train['sex_ Female'])),
                    oxonfair.group_metrics.accuracy(np.asarray(train_y), preds_train,np.asarray(train['sex_ Female'])),
                    label='Train: FairLearn eps='+str(eps), s=50,marker='p')
        plt.scatter(met(np.asarray(test_y),preds,np.asarray(test['sex_ Female'])),
                oxonfair.group_metrics.accuracy(np.asarray(test_y), preds,np.asarray(test['sex_ Female'])),
                label='Test: FairLearn eps='+str(eps), s=50,marker='p')
        plt.scatter(met(np.asarray(val_y),preds_val,np.asarray(val['sex_ Female'])),
                oxonfair.group_metrics.accuracy(np.asarray(val_y), preds_val,np.asarray(val['sex_ Female'])),
                label='Val: FairLearn eps='+str(eps), s=50,marker='p')
    plt.title(classifier_string)
    plt.legend(fontsize=10, ncol=2)
    plt.ylim(bottom=0.65)
    plt.margins(0,0)
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
            hspace = 0, wspace = 0)
    #plt.gca().xaxis.set_major_locator(plt.NullLocator())
    #plt.gca().yaxis.set_major_locator(plt.NullLocator())
    if fair_met=='dp':
        plt.savefig('DemographicParity_'+classifier_string, bbox_inches = 'tight', pad_inches = 0)
    else:
        plt.savefig('EqualOpportunity_'+classifier_string, bbox_inches = 'tight', pad_inches = 0)
