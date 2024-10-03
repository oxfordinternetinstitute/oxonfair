from oxonfair import dataset_loader, DeepFairPredictor, performance
from oxonfair import group_metrics as gm
import numpy as np
from torch import nn, optim, tensor
import torch.nn.functional as F
from numpy import random
from sklearn.preprocessing import OneHotEncoder
import copy
train, _, _ = dataset_loader.compas(train_proportion=1.0, test_proportion=0, groups='race',
                                    replace_groups={'Asian': 'Other', 'Hispanic': 'Other', 'Native American': 'Other'}, seperate_groups=True)


# Define a custom loss that trains the two-heads as required.
def loss(network, x, y, g):
    output = network(x)
    loss0 = F.binary_cross_entropy_with_logits(output[:, 0], y)
    loss1 = F.mse_loss(output[:, 1:], g)
    return loss0 + loss1


def test_1_hot(head_width=4):
    target = tensor(train['target']).float()
    data = tensor(np.asarray(train['data'])).float()

    groups = tensor(OneHotEncoder(sparse_output=False).fit_transform(train['groups'].reshape(-1, 1))).float()
    groups_bin = groups[:, 0]

    std = train['data'].std()
    train['data'] = train['data'] / std

    # define a basic nn with 2 hidden-layers. 1 of width 100, and the second width 50.
    network = nn.Sequential(nn.Linear(train['data'].shape[1], 100),
                            nn.SELU(),
                            nn.Linear(100, 50),
                            nn.SELU(),
                            nn.Linear(50, head_width))
    optimizer = optim.Adam(network.parameters(), lr=1e-4)
    # Train the network
    batch_size = 50

    for epoch in range(100):
        # shuffle data
        perm = random.permutation(target.shape[0])
        target = target[perm]
        data = data[perm]
        groups = groups[perm]
        for step in range(target.shape[0]//batch_size):  # This discards the final incomplete batch
            optimizer.zero_grad()
            el = loss(network,
                      data[step*batch_size:(1+step)*batch_size],
                      target[step*batch_size:(1+step)*batch_size],
                      groups[step*batch_size:(1+step)*batch_size, :head_width-1])
            el.backward()
            optimizer.step()

    train_output = np.asarray(network(tensor(np.asarray(train['data'])).float()).detach())
    if head_width > 2:
        fpred = DeepFairPredictor(train['target'],
                                  train_output,
                                  groups=train['groups'])
    else:
        fpred = DeepFairPredictor(train['target'],
                                  train_output,
                                  groups=groups_bin)

    fpred.fit(gm.accuracy, gm.demographic_parity, 0.01)
    assert fpred.evaluate_fairness()['updated']['Statistical Parity'] < 0.01
    fair_network = copy.deepcopy(network)
    fair_network[-1] = fpred.merge_heads_pytorch(network[-1])
    output_fair = np.asarray(fair_network(tensor(np.asarray(train['data'])).float()).detach())
    if head_width > 2:
        assert (performance.evaluate_fairness(train['target'], output_fair.reshape(-1), train['groups'])
                == fpred.evaluate_fairness()['updated']).all()
        assert np.isclose(performance.evaluate(train['target'], output_fair.reshape(-1)), fpred.evaluate()['updated'], atol=0.001).all()
        assert performance.evaluate_fairness(train['target'], output_fair.reshape(-1), train['groups']).loc['Statistical Parity'] < 0.01
    else:
        assert (performance.evaluate_fairness(train['target'], output_fair.reshape(-1), groups_bin) == fpred.evaluate_fairness()['updated']).all()
        assert np.isclose(performance.evaluate(train['target'], output_fair.reshape(-1)), fpred.evaluate()['updated'], atol=0.001).all()
        assert performance.evaluate_fairness(train['target'], output_fair.reshape(-1), groups_bin).loc['Statistical Parity'] < 0.01


def test_bin():
    test_1_hot(2)
