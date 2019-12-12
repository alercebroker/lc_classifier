import unittest
import pandas as pd
import numpy as np

from late_classifier.features.features import FeaturesComputer
from late_classifier.features.feature_extraction import FeaturesComputerV2

import time


FEATURES_LIST = ['n_samples',
                 'Amplitude',
                 'AndersonDarling',
                 'Autocor_length',
                 'Beyond1Std',
                 'CAR_sigma',
                 'CAR_mean',
                 'CAR_tau',
                 'Con',
                 'Eta_e',
                 'Gskew',
                 'MaxSlope',
                 'Mean',
                 'Meanvariance',
                 'MedianAbsDev',
                 'MedianBRP',
                 'PairSlopeTrend',
                 'PercentAmplitude',
                 'Q31',
                 'PeriodLS',
                 'Period_fit',
                 'Psi_CS',
                 'Psi_eta',
                 'Rcs',
                 'Skew',
                 'SmallKurtosis',
                 'Std',
                 'StetsonK',
                 'Freq1_harmonics_amplitude_0',
                 'Freq1_harmonics_rel_phase_0',
                 'Freq1_harmonics_amplitude_1',
                 'Freq1_harmonics_rel_phase_1',
                 'Freq1_harmonics_amplitude_2',
                 'Freq1_harmonics_rel_phase_2',
                 'Freq1_harmonics_amplitude_3',
                 'Freq1_harmonics_rel_phase_3',
                 'Freq2_harmonics_amplitude_0',
                 'Freq2_harmonics_rel_phase_0',
                 'Freq2_harmonics_amplitude_1',
                 'Freq2_harmonics_rel_phase_1',
                 'Freq2_harmonics_amplitude_2',
                 'Freq2_harmonics_rel_phase_2',
                 'Freq2_harmonics_amplitude_3',
                 'Freq2_harmonics_rel_phase_3',
                 'Freq3_harmonics_amplitude_0',
                 'Freq3_harmonics_rel_phase_0',
                 'Freq3_harmonics_amplitude_1',
                 'Freq3_harmonics_rel_phase_1',
                 'Freq3_harmonics_amplitude_2',
                 'Freq3_harmonics_rel_phase_2',
                 'Freq3_harmonics_amplitude_3',
                 'Freq3_harmonics_rel_phase_3']

FEATURES_LIST = [f+'_1' for f in FEATURES_LIST] + [f+'_2' for f in FEATURES_LIST] + ['gal_b', 'gal_l']


class TestFeaturesComputer(unittest.TestCase):
    def setUp(self):
        self.df_alerts_long = pd.read_pickle('data/object_62alerts.pkl')
        self.df_alerts_three_objects = pd.read_pickle('data/alerts_three_objects.pkl')
        self.features_computer = FeaturesComputer(0)

    def testAllFeaturesComputedLong(self):
        features_df = self.features_computer.execute(self.df_alerts_long)
        for feature in FEATURES_LIST:
            self.assertTrue(feature in features_df.columns)
        self.assertTrue(features_df.shape == (1, len(FEATURES_LIST)))

    def testValidValuesLong(self):
        features_df = self.features_computer.execute(self.df_alerts_long)
        for feature in FEATURES_LIST:
            self.assertTrue(np.isfinite(features_df[feature].values))

    def testShortObject(self):
        features_df = self.features_computer.execute(self.df_alerts_long.head())
        self.assertTrue(features_df.shape == (0, len(FEATURES_LIST)))

    def testCorruptedInput(self):
        with self.assertRaises(ValueError):
            self.features_computer.execute(self.df_alerts_long, filter_corrupted_alerts=False)
        features_df = self.features_computer.execute(self.df_alerts_long)
        for feature in FEATURES_LIST:
            self.assertTrue(np.isfinite(features_df[feature].values))

    def testSingleCorruptedAlert(self):
        single_alert = pd.read_pickle('data/single_corrupted_alert.pkl')
        features_df = self.features_computer.execute(single_alert)
        self.assertTrue(features_df.shape == (0, len(FEATURES_LIST)))

    def testManyObjects(self):
        features_df = self.features_computer.execute(self.df_alerts_three_objects)
        self.assertTrue(features_df.shape == (2, len(FEATURES_LIST)))

    def testManyObjectsOneCorrupted(self):
        second_alert = self.df_alerts_three_objects.iloc[[1]]
        without_first_object = self.df_alerts_three_objects.drop(second_alert.index)
        df = pd.concat((second_alert, without_first_object))
        features_df = self.features_computer.execute(df)
        self.assertTrue(features_df.shape == (1, len(FEATURES_LIST)))

    def testInfiniteValue(self):
        self.df_alerts_long.iloc[5, self.df_alerts_long.columns.get_loc('magpsf_corr')] = np.inf
        self.features_computer.execute(self.df_alerts_long)


class TestFeaturesComputerV2(unittest.TestCase):
    def setUp(self):
        self.features_computer = FeaturesComputerV2(0)

    def testNiceRRL(self):
        df_alerts = pd.read_pickle('data/nice_rrl.pkl')
        features = self.features_computer.execute(df_alerts)
        REAL_PERIOD = 0.409
        periods = features[[
            'PeriodLS_v2_1', 'PeriodLS_v2_2']].values.flatten()
        for period in periods:
            print(period)
            self.assertTrue(np.abs(period-REAL_PERIOD)/REAL_PERIOD < 0.01)

    def testManyZTFCurves(self):
        df_alerts = pd.read_pickle('data/subset.pkl')
        t = time.time()
        features = self.features_computer.execute(df_alerts, verbose=3)
        t = time.time() - t
        print(f'Time many lcs (short): {t/len(features)} in average')

    def testLongZTFCurves(self):
        df_alerts = pd.read_pickle('data/subset_long.pkl')
        t = time.time()
        features = self.features_computer.execute(df_alerts, verbose=3)
        t = time.time() - t
        print(f'Time long lcs: {t/len(features)} in average')
        oid = 'ZTF18aaxjiln'
        true_gal_l = 74.450
        true_gal_b = 2.107
        tol = 1e-2
        computed_gals = features.loc[[oid]][['gal_l', 'gal_b']].values.flatten()
        self.assertTrue(np.abs(true_gal_l - computed_gals[0]) < tol)
        self.assertTrue(np.abs(true_gal_b - computed_gals[1]) < tol)


if __name__ == '__main__':
    unittest.main()
