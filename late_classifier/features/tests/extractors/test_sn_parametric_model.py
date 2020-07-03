import unittest
import os
import numpy as np
import pandas as pd
from late_classifier.features import SNParametricModelExtractor
from late_classifier.features import DetectionsPreprocessorZTF
from late_classifier.features.extractors.sn_model_tf import SNModel


FILE_PATH = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_PATH = os.path.abspath(os.path.join(FILE_PATH, "../data"))


# Just to test SNModel with tensorflow
class SNParametricModelTFExtractor(SNParametricModelExtractor):
    def __init__(self):
        self.sn_model = SNModel()


class TestSNParametricModelExtractor(unittest.TestCase):
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

        self.detections = self.detections[[
            'mjd',
            'fid',
            'magpsf',
            'sigmapsf'
        ]]

    def test_many_objects(self):
        sn_det_extractor = SNParametricModelExtractor()
        sn_det_results = sn_det_extractor.compute_features(self.detections)
        self.assertEqual(
            (4, 2 * len(sn_det_extractor.get_features_keys())),
            sn_det_results.shape)
        print(sn_det_results)
        self.assertEqual(2, len(sn_det_results.dropna()))

    def test_tf_model(self):
        sn_det_extractor = SNParametricModelTFExtractor()
        sn_det_results = sn_det_extractor.compute_features(self.detections)
        self.assertEqual(
            (4, 2 * len(sn_det_extractor.get_features_keys())),
            sn_det_results.shape)
        print(sn_det_results)
        self.assertEqual(2, len(sn_det_results.dropna()))


if __name__ == '__main__':
    unittest.main()
