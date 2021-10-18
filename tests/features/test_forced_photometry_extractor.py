import unittest

import numpy as np
import pandas as pd
from lc_classifier.features import ZTFForcedPhotometryFeatureExtractor
from lc_classifier.features import ZTFLightcurvePreprocessor
from lc_classifier.utils import mag_to_flux


class ForcedPhotometryExtractorTest(unittest.TestCase):
    def setUp(self):
        self.forced_photometry_extractor = ZTFForcedPhotometryFeatureExtractor(
            (1, 2))
        self.detections = pd.read_csv(
            "data_examples/100_objects_detections_corr.csv", index_col="objectId")
        self.detections.index.name = 'oid'

        self.objects = pd.read_csv("data_examples/100_objects.csv", index_col="objectId")
        self.objects.index.name = 'oid'

        self.detections['difference_flux'] = mag_to_flux(self.detections['magpsf'].values)
        self.detections['difference_flux_error'] = (
            mag_to_flux(self.detections['magpsf'].values - self.detections['sigmapsf'].values)
            - self.detections['difference_flux'].values
        )

        self.metadata = self.detections[['ra', 'dec']].groupby(level=0).mean()

    def test_forced_photometry_extractor(self):
        preprocessor = ZTFLightcurvePreprocessor(stream=False)
        detections = preprocessor.preprocess(self.detections, self.objects)

        features_df = self.forced_photometry_extractor.compute_features(
            detections=detections, metadata=self.metadata)

        self.assertEqual(98, features_df.shape[0])
        self.assertEqual(
            len(self.forced_photometry_extractor.get_features_keys()),
            features_df.shape[1])
        self.assertFalse(np.all(np.isnan(features_df.values), axis=(0, 1)))

    def test_get_features_keys(self):
        keys = self.forced_photometry_extractor.get_features_keys()
        self.assertEqual(150, len(keys))


if __name__ == '__main__':
    unittest.main()
