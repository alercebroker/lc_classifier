import unittest
import os
import pandas as pd
from late_classifier.features import ColorFeatureExtractor
from late_classifier.features import DetectionsPreprocessorZTF


FILE_PATH = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_PATH = os.path.abspath(os.path.join(FILE_PATH, "../data"))


class TestColors(unittest.TestCase):
    def setUp(self) -> None:
        fake_objects = pd.DataFrame(
            index=['ZTF17aaaaaxg', 'ZTF18abvvcko', 'ZTF18abakgtm', 'ZTF18aaveorp'],
            data=[[True], [False], [True], [False]],
            columns=['corrected']
        )
        self.objects = fake_objects
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

    def test_many_objects(self):
        color_extractor = ColorFeatureExtractor()
        colors = color_extractor.compute_features(self.detections)
        self.assertEqual(colors.shape, (4, 4))
        objects_without_colors = colors[['g-r_max', 'g-r_mean']].isna().any(axis=1).values.flatten()
        self.assertTrue(
            (objects_without_colors == [False, True, True, False]).all()
        )
        objects_without_colors = colors[['g-r_max_corr', 'g-r_mean_corr']].isna().any(axis=1).values.flatten()
        self.assertTrue(
            (objects_without_colors == [False, True, True, True]).all()
        )


if __name__ == '__main__':
    unittest.main()
