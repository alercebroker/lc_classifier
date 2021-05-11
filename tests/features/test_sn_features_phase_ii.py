import unittest
from lc_classifier.features import SNFeaturesPhaseIIExtractor
from forced_photometry_data.txt_lc_parser import build_forced_detections_df
from lc_classifier.features import ZTFLightcurvePreprocessor


class TestSNFeaturesPhaseII(unittest.TestCase):
    def setUp(self):
        self.extractor = SNFeaturesPhaseIIExtractor([1, 2])
        self.preprocessor = ZTFLightcurvePreprocessor()
        detection_examples = build_forced_detections_df()
        self.detection_examples = self.preprocessor.rename_columns_detections(
            detection_examples)

    def test_something(self):
        features_df = self.extractor.compute_features(
            detections=self.detection_examples
        )
        print(features_df)


if __name__ == '__main__':
    unittest.main()
