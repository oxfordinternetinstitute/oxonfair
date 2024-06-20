from oxonfair import dataset_loader, FairPredictor
from oxonfair import group_metrics as gm
import numpy as np
import xgboost


def test_no_discard():
    'Confirm that predict works on raw data when not discarding groups'
    train_data, val_data, test_data = dataset_loader.adult('sex')
    predictor = xgboost.XGBClassifier().fit(X=train_data['data'], y=train_data['target'])
    fpredict = FairPredictor(predictor, val_data)
    fpredict.fit(gm.accuracy, gm.equal_opportunity, 0.02)
    fpredict.predict(test_data['data'])
    assert True
    # check that specifying groups manualy is the same as not specifying
    fpredict2 = FairPredictor(predictor, val_data, 'sex')
    fpredict2.fit(gm.accuracy, gm.equal_opportunity, 0.02)
    fpredict2.predict(test_data['data'])
    assert np.isclose(fpredict.evaluate_fairness(), fpredict2.evaluate_fairness()).all().all()
    assert True


def test_discard():
    "Confirm that predict doesn't work on raw data when discarding groups"
    train_data, val_data, test_data = dataset_loader.adult('sex', discard_groups=True)
    predictor = xgboost.XGBClassifier().fit(X=train_data['data'], y=train_data['target'])
    fpredict = FairPredictor(predictor, val_data)
    fpredict.fit(gm.accuracy, gm.equal_opportunity, 0.02)
    failed = False
    try:
        fpredict.predict(test_data['data'])
    except AssertionError:
        failed = True
    assert failed


def test_replace():
    train, val, test = dataset_loader.compas('race', train_proportion=0.66, test_proportion=0.33,
                                             discard_groups=True,
                                             replace_groups={'Hispanic': 'Other', 'Native American': ' Other', 'Asian': 'Other'})
    assert (train['groups'] == 'Hispanic').sum() == 0


def test_all_loaders():
    dataset_loader.adult()
    dataset_loader.compas()
    dataset_loader.compas_audit()
    dataset_loader.diabetes()
    dataset_loader.support2()
    dataset_loader.german()
    dataset_loader.taiwan_default()
    dataset_loader.bank_marketing()
    dataset_loader.student()
    dataset_loader.myocardial_infarction()
