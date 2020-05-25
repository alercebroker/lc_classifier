from late_classifier.features import CustomHierarchicalExtractor
from late_classifier.features import DetectionsPreprocessorZTF
import os
import pandas as pd
import unittest


FILE_PATH = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_PATH = os.path.abspath(os.path.join(FILE_PATH, "../data"))


class CustomHierarchicalExtractorTest(unittest.TestCase):
    def setUp(self) -> None:
        preprocess_ztf = DetectionsPreprocessorZTF()

        raw_det_ZTF18abakgtm = pd.read_csv(os.path.join(EXAMPLES_PATH, 'ZTF18abakgtm_det.csv'), index_col="oid")
        det_ZTF18abakgtm = preprocess_ztf.preprocess(raw_det_ZTF18abakgtm)
        raw_nondet_ZTF18abakgtm = pd.read_csv(os.path.join(EXAMPLES_PATH, 'ZTF18abakgtm_nondet.csv'), index_col="oid")

        raw_det_ZTF18abvvcko = pd.read_csv(os.path.join(EXAMPLES_PATH, 'ZTF18abvvcko_det.csv'), index_col="oid")
        det_ZTF18abvvcko = preprocess_ztf.preprocess(raw_det_ZTF18abvvcko)
        raw_nondet_ZTF18abvvcko = pd.read_csv(os.path.join(EXAMPLES_PATH, 'ZTF18abvvcko_nondet.csv'), index_col="oid")

        raw_det_ZTF17aaaaaxg = pd.read_csv(os.path.join(EXAMPLES_PATH, 'ZTF17aaaaaxg_det.csv'), index_col="oid")
        det_ZTF17aaaaaxg = preprocess_ztf.preprocess(raw_det_ZTF17aaaaaxg)
        raw_nondet_ZTF17aaaaaxg = pd.read_csv(os.path.join(EXAMPLES_PATH, 'ZTF17aaaaaxg_nondet.csv'), index_col="oid")

        raw_det_ZTF18aaveorp = pd.read_csv(os.path.join(EXAMPLES_PATH, 'ZTF18aaveorp_det.csv'), index_col="oid")
        det_ZTF18aaveorp = preprocess_ztf.preprocess(raw_det_ZTF18aaveorp)

        keys = [
            'mjd',
            'fid',
            'magpsf_corr',
            'sigmapsf_corr',
            'isdiffpos',
            'magpsf',
            'sigmapsf',
            'ra',
            'dec',
            'sgscore1',
            'rb'
        ]
        self.detections = pd.concat(
            [det_ZTF17aaaaaxg[keys],
             det_ZTF18abvvcko[keys],
             det_ZTF18abakgtm[keys],
             det_ZTF18aaveorp[keys]],
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
            non_detections=self.non_detections
        )
        print(features_df)

    def test_one_object_run(self):
        custom_hierarchical_extractor = CustomHierarchicalExtractor(bands=[1, 2])
        features_df = custom_hierarchical_extractor.compute_features(
            detections=self.detections.loc['ZTF17aaaaaxg'],
            non_detections=self.non_detections.loc['ZTF17aaaaaxg']
        )
        print(features_df)

    def test_just_one_non_detection(self):
        extractor = CustomHierarchicalExtractor(bands=[1, 2])
        results = extractor.compute_features(
            self.detections,
            non_detections=self.just_one_non_detection)

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
            non_detections=self.non_detections
        )
        print(features_df)


if __name__ == '__main__':
    unittest.main()
