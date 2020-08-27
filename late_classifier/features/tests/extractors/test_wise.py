import unittest
import os
import numpy as np
import pandas as pd
from late_classifier.features.extractors import WiseStaticExtractor, WiseStreamExtractor
from late_classifier.features import DetectionsPreprocessorZTF


class WiseStreamExtractorTest(unittest.TestCase):
    def setUp(self):
        self.extractor = WiseStreamExtractor()
        self.detections = pd.DataFrame(
            {"fid": [1, 1, 2, 2], "sigmapsf_corr": [2, 2, 3, 3]}
        )
        self.xmatch = {"W1mag": 1.0, "W2mag": 1.0, "W3mag": 1.0}

    def test_get_features_keys(self):
        features = self.extractor.get_features_keys()
        self.assertEqual(features, ["W1-W2", "W2-W3", "g-W2", "g-W3", "r-W2", "r-W3"])

    def test_calculate_bands(self):
        g, r = self.extractor.calculate_bands(self.detections)
        self.assertEqual(g, 2)
        self.assertEqual(r, 3)

    def test_compute_features_xmatch(self):
        colors = self.extractor.compute_features(self.detections, xmatch=self.xmatch)
        self.assertIsInstance(colors, pd.DataFrame)


FILE_PATH = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_PATH = os.path.abspath(os.path.join(FILE_PATH, "../data"))


class TestWiseExtractor(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(0)

        self.objects = pd.DataFrame(
            index=[
                'ZTF18abakgtm',
                'ZTF18abvvcko',
                'ZTF17aaaaaxg',
                'ZTF18aaveorp'
            ],
            columns=['corrected'],
            data=np.random.choice(
                a=[False, True],
                size=(4,)
            )
        )
        preprocess_ztf = DetectionsPreprocessorZTF()

        raw_det_ZTF18abakgtm = pd.read_csv(
            os.path.join(EXAMPLES_PATH, 'ZTF18abakgtm_det.csv'), index_col="oid")
        raw_det_ZTF18abakgtm['sigmapsf_corr_ext'] = raw_det_ZTF18abakgtm['sigmapsf_corr']

        raw_det_ZTF18abvvcko = pd.read_csv(
            os.path.join(EXAMPLES_PATH, 'ZTF18abvvcko_det.csv'), index_col="oid")
        raw_det_ZTF18abvvcko['sigmapsf_corr_ext'] = raw_det_ZTF18abvvcko['sigmapsf_corr']

        raw_det_ZTF17aaaaaxg = pd.read_csv(
            os.path.join(EXAMPLES_PATH, 'ZTF17aaaaaxg_det.csv'), index_col="oid")
        raw_det_ZTF17aaaaaxg['sigmapsf_corr_ext'] = raw_det_ZTF17aaaaaxg['sigmapsf_corr']

        raw_det_ZTF18aaveorp = pd.read_csv(
            os.path.join(EXAMPLES_PATH, 'ZTF18aaveorp_det.csv'), index_col="oid")
        raw_det_ZTF18aaveorp['sigmapsf_corr_ext'] = raw_det_ZTF18aaveorp['sigmapsf_corr']

        keys = [
            'mjd',
            'fid',
            'magpsf_corr',
            'sigmapsf_corr_ext',
            'isdiffpos',
            'magpsf',
            'sigmapsf',
            'ra',
            'dec',
            'sgscore1',
            'rb'
        ]
        self.detections = pd.concat(
            [raw_det_ZTF17aaaaaxg[keys],
             raw_det_ZTF18abvvcko[keys],
             raw_det_ZTF18abakgtm[keys],
             raw_det_ZTF18aaveorp[keys]],
            axis=0
        )
        self.detections = preprocess_ztf.get_magpsf_ml(
            self.detections,
            objects=self.objects
        )

        self.detections = self.detections[['fid', 'magpsf_ml']]

    def test_many_objects(self):
        wise_extractor = WiseStaticExtractor()
        wise_colors = wise_extractor.compute_features(self.detections)
        self.assertEqual(
            wise_colors.shape,
            (4, len(wise_extractor.get_features_keys())))


if __name__ == '__main__':
    unittest.main()
