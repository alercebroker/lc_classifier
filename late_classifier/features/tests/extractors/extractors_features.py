from late_classifier.features.extractors import *
from late_classifier.features.preprocess import *
import pandas as pd
import unittest

import os

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
print(FILE_PATH)
EXAMPLES_PATH = os.path.abspath(os.path.join(FILE_PATH, "../data"))
print(EXAMPLES_PATH)


class TestObjectsMethods(unittest.TestCase):
    preprocess_ztf = DetectionsPreprocessorZTF()

    raw_det_ZTF18abakgtm = pd.read_csv(os.path.join(EXAMPLES_PATH, 'ZTF18abakgtm_det.csv'), index_col="oid")
    raw_nondet_ZTF18abakgtm = pd.read_csv(os.path.join(EXAMPLES_PATH, 'ZTF18abakgtm_nondet.csv'), index_col="oid")
    det_ZTF18abakgtm = preprocess_ztf.preprocess(raw_det_ZTF18abakgtm)

    raw_det_ZTF18abvvcko = pd.read_csv(os.path.join(EXAMPLES_PATH, 'ZTF18abvvcko_det.csv'), index_col="oid")
    raw_nondet_ZTF18abvvcko = pd.read_csv(os.path.join(EXAMPLES_PATH, 'ZTF18abvvcko_nondet.csv'), index_col="oid")
    det_ZTF18abvvcko = preprocess_ztf.preprocess(raw_det_ZTF18abvvcko)

    def turbofats_features(self):
        turbofats_extractor = TurboFatsFeatureExtractor()
        turbofats_fs = turbofats_extractor.compute_features(self.det_ZTF18abakgtm)
        expected_cols = ['Amplitude', 'AndersonDarling', 'Autocor_length', 'Beyond1Std', 'Con',
                         'Eta_e', 'Gskew', 'MaxSlope', 'Mean', 'Meanvariance', 'MedianAbsDev',
                         'MedianBRP', 'PairSlopeTrend', 'PercentAmplitude', 'Q31', 'PeriodLS_v2',
                         'Period_fit_v2', 'Psi_CS_v2', 'Psi_eta_v2', 'Rcs', 'Skew',
                         'SmallKurtosis', 'Std', 'StetsonK', 'Harmonics_mag_1',
                         'Harmonics_mag_2', 'Harmonics_mag_3', 'Harmonics_mag_4',
                         'Harmonics_mag_5', 'Harmonics_mag_6', 'Harmonics_mag_7',
                         'Harmonics_phase_2', 'Harmonics_phase_3', 'Harmonics_phase_4',
                         'Harmonics_phase_5', 'Harmonics_phase_6', 'Harmonics_phase_7',
                         'Harmonics_mse', 'Pvar', 'ExcessVar', 'GP_DRW_sigma', 'GP_DRW_tau',
                         'SF_ML_amplitude', 'SF_ML_gamma', 'IAR_phi', 'LinearTrend']
        self.assertEqual(type(turbofats_fs), pd.DataFrame)
        self.assertEqual(len(turbofats_fs), 46)
        self.assertListEqual(turbofats_fs.columns, expected_cols)

    def test_color_features(self):
        color_extractor = ColorFeatureExtractor()
        color_fs = color_extractor.compute_features(self.det_ZTF18abvvcko)
        self.assertEqual(type(color_fs), pd.DataFrame)
        self.assertEqual(len(color_fs.columns), 2)
        self.assertListEqual(list(color_fs.columns), color_extractor.features_keys)

    def test_galactic_coordinates_features(self):
        galactic_extractor = GalacticCoordinatesComputer()
        galactic_fs = galactic_extractor.compute_features(self.det_ZTF18abakgtm)
        self.assertEqual(type(galactic_fs), pd.DataFrame)
        self.assertEqual(len(galactic_fs.columns), 2)
        self.assertListEqual(list(galactic_fs.columns), galactic_extractor.features_keys)

    def test_real_bogus_features(self):
        rb_extractor = RealBogusComputer()
        rb_fs = rb_extractor.compute_features(self.det_ZTF18abakgtm)
        self.assertEqual(type(rb_fs), pd.DataFrame)

    def test_sg_score_features(self):
        sg_score = SGScoreComputer()
        sg_fs = sg_score.compute_features(self.det_ZTF18abakgtm)
        self.assertEqual(type(sg_fs), pd.DataFrame)

    def test_sn_detections_features(self):
        sn_det_extractor = SupernovaeDetectionFeatureComputer()
        sn_fs = sn_det_extractor.compute_features(self.det_ZTF18abakgtm)
        self.assertEqual(type(sn_fs), pd.DataFrame)

    def test_sn_non_detections_features(self):
        sn_non_det_extractor = SupernovaeNonDetectionFeatureComputer()
        sn_non_det = sn_non_det_extractor.compute_features(self.det_ZTF18abakgtm, non_detections=self.raw_nondet_ZTF18abakgtm)
        self.assertEqual(type(sn_non_det), pd.DataFrame)

    def test_hierarchical_features(self):
        hierarchical_extractor = HierarchicalFeaturesComputer([1, 2])
        h_fs = hierarchical_extractor.compute_features(self.det_ZTF18abakgtm, non_detections=self.raw_nondet_ZTF18abakgtm)
        self.assertEqual(type(h_fs), pd.DataFrame)


if __name__ == '__main__':
    unittest.main()
