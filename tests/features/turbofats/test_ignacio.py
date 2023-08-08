import unittest
from lc_classifier.features.turbofats.FeatureFunctionLib import Mean
from lc_classifier.features.turbofats import FeatureSpace
import numpy as np
import pandas as pd
import os


class TestFeatures(unittest.TestCase):
    def setUp(self):
        this_dir = os.path.dirname(os.path.abspath(__file__))
        lc_filename = os.path.join(
            this_dir,
            "ZTF18aaiopei_detections.pkl"
        )
        self.lc_data = pd.read_pickle(lc_filename)
        self.lc_g = self.lc_data[self.lc_data.fid == 1]
        self.lc_r = self.lc_data[self.lc_data.fid == 2]
        self.lc_g_np = self.lc_g[['magpsf_corr', 'mjd', 'sigmapsf_corr']].values.T

    def testMean(self):
        mean_computer = Mean(shared_data=None)
        mean = mean_computer.fit(self.lc_g_np)
        expected_mean = 17.371986
        self.assertTrue(np.abs(mean - expected_mean) < 0.01)


class TestNewFeatureSpace(unittest.TestCase):
    def setUp(self):
        this_dir = os.path.dirname(os.path.abspath(__file__))
        lc_filename = os.path.join(
            this_dir,
            "ZTF18aaiopei_detections.pkl"
        )
        self.lc_data = pd.read_pickle(lc_filename)
        self.lc_g = self.lc_data[self.lc_data.fid == 1]
        self.lc_r = self.lc_data[self.lc_data.fid == 2]
        self.lc_g_np = self.lc_g[['magpsf_corr', 'mjd', 'sigmapsf_corr']].values.T

    def test1(self):
        feature_list = [
            'Amplitude', 'AndersonDarling', 'Autocor_length',
            'Beyond1Std',
            'Con', 'Eta_e',
            'Gskew',
            'MaxSlope', 'Mean', 'Meanvariance', 'MedianAbsDev',
            'MedianBRP', 'PairSlopeTrend', 'PercentAmplitude', 'Q31',
            'Psi_CS_v2', 'Psi_eta_v2', 'Rcs',
            'Skew', 'SmallKurtosis', 'Std',
            'StetsonK',
            'Pvar', 'ExcessVar',
            'SF_ML_amplitude', 'SF_ML_gamma',
            'IAR_phi',
            'LinearTrend'
        ]
        new_feature_space = FeatureSpace(
            feature_list,
            data_column_names=['magpsf_corr', 'mjd', 'sigmapsf_corr']
        )
        features = new_feature_space.calculate_features(self.lc_g)
        print(features.iloc[0])
        expected_mean = 17.371986
        mean = features['Mean'][0]
        self.assertTrue(np.abs(mean - expected_mean) < 0.01)

if __name__ == '__main__':
    unittest.main()
