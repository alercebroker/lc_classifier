from late_classifier.features import CustomHierarchicalExtractor
import os
import numpy as np
import pandas as pd
import unittest


FILE_PATH = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_PATH = os.path.abspath(os.path.join(FILE_PATH, "../data"))


class CustomHierarchicalExtractorTest(unittest.TestCase):
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

        raw_det_ZTF18abakgtm = pd.read_csv(
            os.path.join(EXAMPLES_PATH, 'ZTF18abakgtm_det.csv'), index_col="oid")
        raw_det_ZTF18abakgtm['sigmapsf_corr_ext'] = raw_det_ZTF18abakgtm['sigmapsf_corr']
        raw_nondet_ZTF18abakgtm = pd.read_csv(
            os.path.join(EXAMPLES_PATH, 'ZTF18abakgtm_nondet.csv'), index_col="oid")

        raw_det_ZTF18abvvcko = pd.read_csv(
            os.path.join(EXAMPLES_PATH, 'ZTF18abvvcko_det.csv'), index_col="oid")
        raw_det_ZTF18abvvcko['sigmapsf_corr_ext'] = raw_det_ZTF18abvvcko['sigmapsf_corr']
        raw_nondet_ZTF18abvvcko = pd.read_csv(
            os.path.join(EXAMPLES_PATH, 'ZTF18abvvcko_nondet.csv'), index_col="oid")

        raw_det_ZTF17aaaaaxg = pd.read_csv(
            os.path.join(EXAMPLES_PATH, 'ZTF17aaaaaxg_det.csv'), index_col="oid")
        raw_det_ZTF17aaaaaxg['sigmapsf_corr_ext'] = raw_det_ZTF17aaaaaxg['sigmapsf_corr']
        raw_nondet_ZTF17aaaaaxg = pd.read_csv(
            os.path.join(EXAMPLES_PATH, 'ZTF17aaaaaxg_nondet.csv'), index_col="oid")

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

        non_det_keys = ["diffmaglim", "mjd", "fid"]
        self.non_detections = pd.concat(
            [
                raw_nondet_ZTF17aaaaaxg[non_det_keys],
                raw_nondet_ZTF18abakgtm[non_det_keys],
                raw_nondet_ZTF18abvvcko[non_det_keys]
            ],
            axis=0
        )
        self.just_one_non_detection = raw_nondet_ZTF17aaaaaxg[non_det_keys].iloc[[0]]

    def test_if_runs_many_objects(self):
        custom_hierarchical_extractor = CustomHierarchicalExtractor(bands=[1, 2])
        features_df = custom_hierarchical_extractor.compute_features(
            detections=self.detections,
            non_detections=self.non_detections,
            objects=self.objects
        )
        print(features_df)

    def test_one_object_run(self):
        custom_hierarchical_extractor = CustomHierarchicalExtractor(bands=[1, 2])
        features_df = custom_hierarchical_extractor.compute_features(
            detections=self.detections.loc['ZTF17aaaaaxg'],
            non_detections=self.non_detections.loc['ZTF17aaaaaxg'],
            objects=self.objects
        )
        print(features_df)

    def test_just_one_non_detection(self):
        extractor = CustomHierarchicalExtractor(bands=[1, 2])
        results = extractor.compute_features(
            self.detections,
            non_detections=self.just_one_non_detection,
            objects=self.objects
        )

    def test_object_with_few_alerts(self):
        custom_hierarchical_extractor = CustomHierarchicalExtractor(bands=[1, 2])
        oids = self.detections.index.unique()
        detections = self.detections.loc[oids[:3]]
        last_curve_oid = oids[-1]
        last_curve = self.detections.loc[last_curve_oid]
        last_curve_short = last_curve.iloc[:4]
        detections = pd.concat([detections, last_curve_short], axis=0)
        features_df = custom_hierarchical_extractor.compute_features(
            detections=detections,
            non_detections=self.non_detections,
            objects=self.objects
        )
        print(features_df)


if __name__ == '__main__':
    unittest.main()
