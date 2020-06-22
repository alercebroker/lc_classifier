import unittest

import pandas as pd
from late_classifier.features import DetectionsPreprocessorZTF


class TestDetectionsPreprocessor(unittest.TestCase):
    def setUp(self) -> None:
        self.detections = pd.read_pickle('data/test_detections.pkl')

    def test_non_numeric_columns(self):
        # print(self.detections.head())
        # print(self.detections.dtypes)
        with self.assertRaises(TypeError):
            _ = self.detections.sigmapsf_corr > 0.0

        preprocessor = DetectionsPreprocessorZTF()
        preprocessed_detections = preprocessor.preprocess_detections(self.detections)
