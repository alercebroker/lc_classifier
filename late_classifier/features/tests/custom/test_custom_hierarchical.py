from late_classifier.features.custom import CustomHierarchicalExtractor
import os
import pandas as pd
import unittest


FILE_PATH = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_PATH = os.path.abspath(os.path.join(FILE_PATH, "../data"))


class CustomHierarchicalExtractorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.detections = pd.read_csv(
            os.path.join(EXAMPLES_PATH, 'ZTF18aazsabq_det.csv'),
            index_col='oid'
        )
        self.non_detections = pd.read_csv(
            os.path.join(EXAMPLES_PATH, 'ZTF18aazsabq_nondet.csv'),
            index_col='oid'
        )

    def test_if_runs(self):
        custom_hierarchical_extractor = CustomHierarchicalExtractor(bands=[1, 2])
        features_df = custom_hierarchical_extractor.compute_features(
            detections=self.detections,
            non_detections=self.non_detections
        )
        is_nan_feature = pd.isna(features_df).values.any()
        self.assertFalse(is_nan_feature)


if __name__ == '__main__':
    unittest.main()
