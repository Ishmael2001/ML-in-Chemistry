#!/usr/bin/env python
#coding=utf-8

"""
Author:        Pengbo Song
Created Date:  2021/10/11
Last Modified: 2021/10/11
"""

import os
import unittest

# Test package dependency
try:
    import numpy as np
except:
    raise ModuleNotFoundError("Can not import numpy module.")

nptesting = False
try:
    from numpy import testing
except:
    raise ModuleNotFoundError("Can not import numpy testing module.")
finally:
    nptesting = True

try:
    from sklearn import linear_model
except:
    raise ModuleNotFoundError("Can not import scikit-learn linear-model module.")

try:
    from sklearn import metrics
except:
    raise ModuleNotFoundError("Can not import scikit-learn metrcis module.")

# Test target code availability
workdir = os.path.dirname(__file__)
targetfnm = "ridge_regression.py"
if not os.path.isfile(os.path.join(workdir, targetfnm)):
    raise OSError(f"Can not find {targetfnm} in current workdir."
                   "Please confirm this code file is placed in the correct directory.")
try:
    from ridge_regression import MLChemLab1
except:
    raise ModuleNotFoundError("Can not load class MLChemLab1 from target code file. Test failed.")

# Test data file availability
datadir = os.path.join(workdir, "..", "data")
trainfnm = "train.dat"
testfnm = "test.dat"
trainfpath = os.path.join(datadir, trainfnm)
testfpath = os.path.join(datadir, testfnm)
if not (os.path.isfile(trainfpath) and
        os.path.isfile(testfpath)):
    raise OSError(f"Can not find {trainfnm}, {testfnm} in data directory.")


def parse_dat(filenm: str):
    """Parse DAT file and pack data into numpy float array"""
    xarr = []; yarr = []
    with open(filenm, 'r') as f:
        for line in f.readlines():
            # Read lines in DAT file and split each line by space
            x, y = line.strip().split(' ')
            # Convert splitted values to float and append to the end of list
            xarr.append(float(x))
            yarr.append(float(y))
    # Convert list to numpy array
    xarr = np.asarray(xarr, dtype=float)
    yarr = np.asarray(yarr, dtype=float)
    assert xarr.size == yarr.size, "Got unequal length of X and y vector"
    # Returns array extracted from DAT file
    return xarr, yarr


class TestLab1(unittest.TestCase):
    """Test functionalities of methods and models in MLChemLab1"""
    def setUp(self):
        """Prepare test environment"""
        self.train_X, self.train_y = parse_dat(trainfpath)
        self.test_X, self.test_y = parse_dat(testfpath)

    def test_data(self):
        """Test raw data file is handled correctly"""
        self.assertEqual(self.train_X.dtype, float, msg="Train X should be a float array.")
        self.assertEqual(self.train_X.size, 12, msg="Train X should have 12 elements.")
        self.assertEqual(self.train_y.dtype, float, msg="Train y should be a float array.")
        self.assertEqual(self.train_y.size, 12, msg="Train y should have 12 elements.")
        self.assertEqual(self.test_X.dtype, float, msg="Test X should be a float array.")
        self.assertEqual(self.test_X.size, 300, msg="Test X should have 300 elements.")
        self.assertEqual(self.test_y.dtype, float, msg="Test y should be a float array.")
        self.assertEqual(self.test_y.size, 300, msg="Test y should have 300 elements.")

    def test_ridge(self):
        """Test ridge regression models

        Test methods:
            add_model(model : str, alpha : float)
            fit(x : np.ndarray[2d, float], y : np.ndarray[1d, float], featurization_mode : str, degree : int)
            predict(x : np.ndarray[2d, float])
            evaluation(y_predict : np.ndarray[1d, float], y_label : np.ndarray[1d, float], metric : str)
        
        Detailed procedures:
        1. Initialize ridge regression model, set regularization factor (alpha / lambda) to 1.0
        2. Fit model with X and y from train dataset, with featurization mode set to poly and degree set to 4
        3. Check whether predict y is all close to true y
        4. Check whether calculated RMSE (Root Mean Squared Error) is close to the right value
        5. Fit model with X and y from train dataset, with featurization mode set to poly-cos and degree set to 4
        6. Check whether predict y is all close to true y
        7. Check whether calculated RMSE (Root Mean Squared Error) is close to the right value
        """
        ridge_model = MLChemLab1()
        ridge_model.add_model("ridge", alpha=1.0)
        ridge_model.fit(self.train_X, self.train_y, featurization_mode="poly", degree=4)
        test_y_pred = ridge_model.predict(self.test_X)
        self.assertEqual(test_y_pred.dtype, float, msg="Predict test y should be a float array.")
        self.assertEqual(test_y_pred.size, 300, msg="Predict test y should have 300 elements.")
        if nptesting:
            testing.assert_allclose(self.test_y, test_y_pred, atol=50.0)
        rmse = metrics.mean_squared_error(self.test_y, test_y_pred, squared=False)
        eval_rmse = ridge_model.evaluation(test_y_pred, self.test_y, metric="RMS")
        self.assertAlmostEqual(rmse, eval_rmse, places=1)
        
        ridge_model.fit(self.train_X, self.train_y, featurization_mode="poly-cos", degree=4)
        test_y_pred = ridge_model.predict(self.test_X)
        self.assertEqual(test_y_pred.dtype, float, msg="Predict test y should be a float array.")
        self.assertEqual(test_y_pred.size, 300, msg="Predict test y should have 300 elements.")
        if nptesting:
            testing.assert_allclose(self.test_y, test_y_pred, atol=20.0)
        rmse = metrics.mean_squared_error(self.test_y, test_y_pred, squared=False)
        eval_rmse = ridge_model.evaluation(test_y_pred, self.test_y, metric="RMS")
        self.assertAlmostEqual(rmse, eval_rmse, places=1)
    
    def test_naive(self):
        """Test naive model

        Test methods:
            add_model(model : str, alpha : float)
            fit(x : np.ndarray[2d, float], y : np.ndarray[1d, float], featurization_mode : str, degree : int)
            predict(x : np.ndarray[2d, float])
            evaluation(y_predict : np.ndarray[1d, float], y_label : np.ndarray[1d, float], metric : str)
        
        Detailed procedures:
        1. Initialize ridge regression model, set regularization factor (alpha / lambda) to 1.0
        2. Fit model with X and y from train dataset, with featurization mode set to identical(*)
        3. Check whether predict y is all close to true y
        4. Check whether calculated RMSE (Root Mean Squared Error) is close to the right value

        (*) Input X data should be handled preperly instead of using raw X data
        """
        naive_model = MLChemLab1()
        naive_model.add_model("naive", alpha=1.0)
        naive_model.fit(self.train_X, self.train_y, featurization_mode="identical")
        test_y_pred = naive_model.predict(self.test_X)
        self.assertEqual(test_y_pred.dtype, float, msg="Predict test y should be a float array.")
        self.assertEqual(test_y_pred.size, 300, msg="Predict test y should have 300 elements.")
        if nptesting:
            testing.assert_allclose(self.test_y, test_y_pred, atol=20.0)
        rmse = metrics.mean_squared_error(self.test_y, test_y_pred, squared=False)
        eval_rmse = naive_model.evaluation(test_y_pred, self.test_y, metric="RMS")
        self.assertAlmostEqual(rmse, eval_rmse, places=1)

if __name__ == '__main__':
    unittest.main(verbosity=1)
