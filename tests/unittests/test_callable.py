import numpy as np 
import oxonfair
from oxonfair.utils import group_metrics as gm


val=np.random.randn(1000,2) #np.load('outputs_val.npy')
val_target = np.random.randn(1000)>0 #np.load('target_labels_val.npy')
val_groups = np.random.randn(1000)>0 #np.load('protected_labels_val.npy')

test=np.random.randn(1001,2) #np.load('outputs_test.npy')
test_target=np.random.randn(1001)>0 #np.load('target_labels_test.npy')
test_groups=np.random.randn(1001)>0 #np.load('protected_labels_test.npy')

def sigmoid(array):
    return np.stack ((1/(1+np.exp(array[:,0])),1/(1+np.exp(-array[:,0]))),1)
def square_align(array):
    return np.stack((array[:,1],0.5-array[:,1]),1)

val_dict={'data':val, 'target':val_target, 'groups':val_groups}
test_dict={'data':test, 'target':test_target, 'groups':test_groups}

def test_runs():
    fpred=oxonfair.FairPredictor(sigmoid, val_dict, val_groups, inferred_groups=square_align)
    fpred.fit(gm.accuracy,gm.equal_opportunity,0.02)
    fpred.plot_frontier()
    fpred.plot_frontier(test_dict)
    fpred.evaluate()
    fpred.evaluate(test_dict)
    fpred.evaluate_fairness()
    fpred.evaluate_fairness(test_dict)
    fpred.evaluate_groups()
    fpred.evaluate_groups(test_dict)
    

