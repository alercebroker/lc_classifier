import unittest
import pandas as pd

from late_classifier.features.preprocess import StreamDetectionsPreprocessorZTF


class TestStreamDetectionsPreprocessorZTF(unittest.TestCase):
    def setUp(self) -> None:
        self.preprocessor_ztf = StreamDetectionsPreprocessorZTF()

        self.detections = pd.read_csv("data_examples/100_objects_detections_corr.csv", index_col="objectId")
        self.detections.index.name = 'oid'

    def test_preprocess(self):
        self.assertEqual(self.detections.index.name, "oid")
        preprocessed_data = self.preprocessor_ztf.preprocess(self.detections)
        self.assertIsInstance(preprocessed_data, pd.DataFrame)
