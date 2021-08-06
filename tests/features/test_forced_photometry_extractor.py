import unittest

import numpy as np
import pandas as pd
from lc_classifier.features import ZTFForcedPhotometryFeatureExtractor
from lc_classifier.features import ZTFLightcurvePreprocessor
from lc_classifier.utils import mag_to_flux


class ForcedPhotometryExtractorTest(unittest.TestCase):
    def setUp(self):
        self.forced_photometry_extractor = ZTFForcedPhotometryFeatureExtractor(
            (1, 2), stream=False)
        self.streamed_forced_photometry_extractor = ZTFForcedPhotometryFeatureExtractor(
            (1, 2), stream=True)
        self.detections = pd.read_csv(
            "data_examples/100_objects_detections_corr.csv", index_col="objectId")
        self.detections.index.name = 'oid'

        self.objects = pd.read_csv("data_examples/100_objects.csv", index_col="objectId")
        self.objects.index.name = 'oid'

        self.detections['diff_flux'] = mag_to_flux(self.detections['magpsf'].values)
        self.detections['diff_err'] = (
            mag_to_flux(self.detections['magpsf'].values - self.detections['sigmapsf'].values)
            - self.detections['diff_flux'].values
        )

    def test_forced_photometry_extractor(self):
        preprocessor = ZTFLightcurvePreprocessor(stream=False)
        detections = preprocessor.preprocess(self.detections, self.objects)

        features_df = self.forced_photometry_extractor.compute_features(
            detections=detections)

        self.assertEqual(98, features_df.shape[0])
        self.assertEqual(
            len(self.forced_photometry_extractor.get_features_keys()),
            features_df.shape[1])
        self.assertFalse(np.all(np.isnan(features_df.values), axis=(0, 1)))

    def test_forced_photometry_extractor_stream(self):
        oid = "ZTF17aaaorfd"
        detections = self.detections.loc[[oid]]
        detections['corrected'] = True
        xmatches = pd.DataFrame(
            data=[[1.0, 1.0, 1.0, 'ZTF17aaaorfd']],
            columns=['W1mag', 'W2mag', 'W3mag', 'oid'])

        metadata = pd.DataFrame(
            data=[['ZTF17aaaorfd', 'fake_candid', 1.0]],
            columns=['oid', 'candid', 'sgscore1'])

        preprocessor = ZTFLightcurvePreprocessor(stream=True)
        detections = preprocessor.preprocess(detections)

        features_df = self.streamed_forced_photometry_extractor.compute_features(
            detections=detections,
            xmatches=xmatches,
            metadata=metadata)
        self.assertEqual(len(features_df), 1)

    def test_get_features_keys(self):
        keys = self.forced_photometry_extractor.get_features_keys()
        keys_stream = self.streamed_forced_photometry_extractor.get_features_keys()
        self.assertEqual(154, len(keys))
        self.assertEqual(154, len(keys_stream))


if __name__ == '__main__':
    unittest.main()
