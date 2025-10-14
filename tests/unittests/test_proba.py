"""Test that validation data is not modified during evaluation"""

import numpy as np
import copy
from oxonfair import FairPredictor
from oxonfair import group_metrics as gm


def test_validation_data_immutable():
    """Test that validation data does not change after evaluation.

    This test verifies that the deepcopy protection in FairPredictor.predict_proba
    prevents modification of the original validation data during evaluation.
    """
    # Prep example data
    n_samples = 50
    target = np.random.randint(0, 2, size=n_samples)
    pred_prob = np.random.rand(n_samples)
    groups = np.random.randint(0, 2, size=n_samples)

    # Convert to correct format
    data_dict = {
        "target": target,
        "data": np.array((1 - pred_prob, pred_prob)).T,
        "groups": groups
    }

    # Create deep copy to compare against later
    original_data_dict = copy.deepcopy(data_dict)

    # Get and train the threshold calibrator
    fpred = FairPredictor(predictor=None, validation_data=data_dict)
    fpred.fit(gm.accuracy, gm.true_pos_rate.diff, 0.05)

    fpred.evaluate_fairness()
    # Check that original data is unchanged after calling fit + evaluation
    np.testing.assert_array_equal(data_dict["target"], original_data_dict["target"])
    np.testing.assert_array_equal(data_dict["data"], original_data_dict["data"])
    np.testing.assert_array_equal(data_dict["groups"], original_data_dict["groups"])

    fpred.predict_proba(data_dict)
    # Check that original data is unchanged after calling predict_proba
    np.testing.assert_array_equal(data_dict["target"], original_data_dict["target"])
    np.testing.assert_array_equal(data_dict["data"], original_data_dict["data"])
    np.testing.assert_array_equal(data_dict["groups"], original_data_dict["groups"])

    fpred.predict(data_dict)
    # Check that original data is unchanged after calling predict
    np.testing.assert_array_equal(data_dict["target"], original_data_dict["target"])
    np.testing.assert_array_equal(data_dict["data"], original_data_dict["data"])
    np.testing.assert_array_equal(data_dict["groups"], original_data_dict["groups"])
