import unittest
import pandas as pd
from lc_classifier.features import ForcedPhotometryExtractor, StreamedForcedPhotometryExtractor


class ForcedPhotometryExtractorTest(unittest.TestCase):
    def setUp(self):
        self.forced_photometry_extractor = ForcedPhotometryExtractor()
        self.streamed_forced_photometry_extractor = StreamedForcedPhotometryExtractor()

        self.detections = pd.read_csv(
            "data_examples/100_objects_detections_corr.csv", index_col="objectId")
        self.detections.index.name = 'oid'

        self.objects = pd.read_csv("data_examples/100_objects.csv", index_col="objectId")
        self.objects.index.name = 'oid'

    def test_custom_hierarchical_features(self):
        features_df = self.forced_photometry_extractor.compute_features(
            detections=self.detections,
            objects=self.objects
        )
        self.assertEqual(features_df.shape[0], 98)
        self.assertEqual(features_df.shape[1], 152)

    def test_custom_hierarchical_features_stream(self):
        oid = "ZTF17aaaorfd"
        detections = self.detections.loc[[oid]]
        xmatches = pd.DataFrame(
            data=[[1.0, 1.0, 1.0, 'ZTF17aaaorfd']],
            columns=['W1mag', 'W2mag', 'W3mag', 'oid'])

        metadata = pd.DataFrame(
            data=[['ZTF17aaaorfd', 'fake_candid', 1.0]],
            columns=['oid', 'candid', 'sgscore1'])

        features_df = self.streamed_forced_photometry_extractor.compute_features(
            detections=detections,
            xmatches=xmatches,
            metadata=metadata)
        self.assertEqual(len(features_df), 1)

    def test_get_features_keys(self):
        keys = self.forced_photometry_extractor.get_features_keys()
        keys_stream = self.streamed_forced_photometry_extractor.get_features_keys()
        self.assertEqual(len(keys), 22)
        self.assertEqual(len(keys_stream), 23)


if __name__ == '__main__':
    unittest.main()
