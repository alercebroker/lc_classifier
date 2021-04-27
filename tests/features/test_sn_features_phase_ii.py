import pandas as pd
import unittest
from lc_classifier.features.extractors.sn_features_phase_ii import SNFeaturesPhaseIIExtractor
from forced_photometry_data.txt_lc_parser import build_forced_detections_df


class TestSNFeaturesPhaseII(unittest.TestCase):
    def setUp(self):
        self.extractor = SNFeaturesPhaseIIExtractor()
        self.detection_examples = build_forced_detections_df()

    def test_something(self):
        features_df = self.extractor.compute_features(
            detections=self.detection_examples
        )
        print(features_df)


if __name__ == '__main__':
    unittest.main()
