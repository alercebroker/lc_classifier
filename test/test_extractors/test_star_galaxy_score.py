import unittest
import pandas as pd
from late_classifier.features.extractors import StreamSGScoreExtractor

class TestStreamSGScore(unittest.TestCase):
    def setUp(self) -> None:
        self.extractor = StreamSGScoreExtractor()

    def test_compute_features(self):
        metadata = {"ps1": {"sgscore": 1.0}}
        sgscore = self.extractor.compute_features(pd.DataFrame(), metadata=metadata)
        self.assertIsInstance(sgscore, pd.DataFrame)
