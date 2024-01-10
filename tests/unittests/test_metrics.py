import numpy as np
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.core import metrics
from oxonfair.utils import group_metrics as gm
from oxonfair.utils import conditional_group_metrics as cgm

def test_metrics():
    "check that core.metrics give the same answer as group metrics"
    array1 = np.random.randint(0, 2, 100)
    array2 = np.random.randint(0, 2, 100)
    array3 = np.zeros(100)
    met_list = (metrics.accuracy,
                metrics.balanced_accuracy,
                metrics.f1,
                metrics.mcc,
                metrics.precision,
                metrics.recall)
    group_met_list = (
        gm.accuracy,
        gm.balanced_accuracy,
        gm.f1,
        gm.mcc,
        gm.precision,
        gm.recall)
    for met, group_met in zip(met_list, group_met_list):
        assert np.isclose(met(array1, array2), group_met(array1, array2, array3)[0], 1e-5)


def test_metrics_identities():
    """ validity check, make sure metrics are consistent with standard identities.
     This combined with test metrics gives coverage of everything up to the clarify metrics"""
    array1 = np.random.randint(0, 2, 100)
    array2 = np.random.randint(0, 2, 100)
    array3 = np.random.randint(0, 4, 100)
    assert np.isclose(gm.pos_data_rate(array1, array2, array3),
                      1 - gm.neg_data_rate(array1, array2, array3)).all()
    assert np.isclose(gm.pos_pred_rate(array1, array2, array3),
                      1 - gm.neg_pred_rate(array1, array2, array3)).all()
    assert np.isclose(gm.true_pos_rate(array1, array2, array3),
                      1 - gm.false_neg_rate(array1, array2, array3)).all()
    assert np.isclose(gm.true_neg_rate(array1, array2, array3),
                      1 - gm.false_pos_rate(array1, array2, array3)).all()
    accuracy = gm.Utility([1, 0, 0, 1], 'accuracy')
    assert np.isclose(gm.accuracy(array1, array2, array3), accuracy(array1, array2, array3)).all()
    # assert np.isclose(gm.(A,B,array3),1-gm.(A,B,array3)).all()
    # check that additive_metrics can be called.
    assert np.isclose(gm.equalized_odds(array1, array2, array3),
                      (gm.true_pos_rate.diff(array1, array2, array3)
                       + gm.true_neg_rate.diff(array1, array2, array3)) / 2).all()

def test_conditional_metrics():
    """Check that conditional metrics with weight 1 are the same as unweighted variants"""
    array1 = np.random.randint(0, 2, 100)
    array2 = np.random.randint(0, 2, 100)
    array3 = np.random.randint(0, 4, 100)
    array4 = np.random.randint(0, 5, 100)
    array5 =np.ones(100)
    cpdr=gm.pos_pred_rate.clone('cpdr', cgm.cond_debug)
    assert np.isclose(gm.pos_pred_rate(array1, array2, array3),
                      cpdr(array1,array2,array3,array4))
    assert cgm.reweight_by_factor_postives(1,1,1,1)==1
    assert cgm.reweight_by_factor_postives(1,0,1,1)==1
    assert cgm.reweight_by_factor_postives(1,0,0,1)==0

    assert np.isclose(gm.pos_pred_rate(array1, array2, array3),
                     cgm.cond_pos_pred_rate(array1,array2,array3,array5))
    assert not np.isclose(cgm.cond_pos_pred_rate(array1,array2,array3,array4),
                           cpdr(array1,array2,array3,array4))

def make_rows(department,gender,accept,count):
    dep = np.empty(count,dtype=str)
    dep[:] = department
    gen = np.empty_like(dep)
    gen[:] = gender
    acc = np.empty(count)
    acc[:] = accept
    pred = np.empty_like(acc)
    pred[:] = np.random.binomial(1,0.5)
    d= {'Department':dep,'Gender':gen,'Accept':acc,'Prediction':pred}
    return pd.DataFrame(d)

def test_conditional_is_consistent():
    "Check that the numbers match those reported in statistics"
    import os
    condensed=pd.read_csv('benchmark/Berkeley.tsv',sep='\t')
    collect = list()
    for idx in range(6):
        collect.append(make_rows(condensed['Department'][idx],'Male',1,condensed['MaleYes'][idx]))
        collect.append(make_rows(condensed['Department'][idx],'Female',1,condensed['FemaleYes'][idx]))
        collect.append(make_rows(condensed['Department'][idx],'Female',0,condensed['FemaleNo'][idx]))
        collect.append(make_rows(condensed['Department'][idx],'Male',0,condensed['MaleNo'][idx]))

    complete=pd.concat(collect)
    pdr = gm.pos_pred_rate.per_group(complete['Prediction'],complete['Accept'],complete['Gender'])
    assert np.isclose(pdr[0,0],0.30,atol=0.01)# Bug in original book
    assert np.isclose(pdr[0,1],0.44,atol=0.01)
    cpdr = cgm.cond_pos_pred_rate.per_group(complete['Prediction'],complete['Accept'],complete['Gender'], complete['Department'])
    assert np.isclose(cpdr[0,0],0.43,atol=0.01)
    assert np.isclose(cpdr[0,1],0.38,atol=0.01)
    

