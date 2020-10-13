import unittest
import pandas as pd
from late_classifier.features.extractors import StreamSGScoreExtractor


class TestStreamSGScore(unittest.TestCase):
    def setUp(self) -> None:
        self.extractor = StreamSGScoreExtractor()

    def test_compute_features(self):
        metadata = {"ps1": {"sgscore1": 1.0}}
        sgscore = self.extractor.compute_features(pd.DataFrame(
            {"oid": ["1", "4", "2", "3"], "something": [2, 2, 3, 3]}
        ), metadata=metadata)
        self.assertIsInstance(sgscore, pd.DataFrame)
