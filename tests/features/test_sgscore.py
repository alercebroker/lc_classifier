import unittest
import pandas as pd
from lc_classifier.features.extractors import StreamSGScoreExtractor


class TestStreamSGScore(unittest.TestCase):
    def setUp(self) -> None:
        self.extractor = StreamSGScoreExtractor()

    def test_compute_features(self):
        metadata = pd.DataFrame(
            {
                'oid': ['1', '2', '3', '4'],
                'candid': ['fake_candid']*4,
                'sgscore1': [1.0]*4
            })
        
        sgscore = self.extractor.compute_features(pd.DataFrame(
            {"oid": ["1", "4", "2", "3"], "something": [2, 2, 3, 3]}
        ).set_index('oid'), metadata=metadata)
        self.assertIsInstance(sgscore, pd.DataFrame)
