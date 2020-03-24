import pandas as pd


class GenericExtractorTest:
    extractor = None
    params = {}

    def test_compute_features(self):
        ext = self.extractor()
        with open(self.params["TEST_DETECTION"], "r") as detections_file:
            detections = pd.read_csv(detections_file)

        try:
            with open(self.params["TEST_NON_DETECTION"], "r") as non_detections_file:
                non_detections = pd.read_csv(non_detections_file)
        except Exception as e:
            non_detections = []

        features = ext.compute_features(detections, non_detections=non_detections)

        self.assertEqual(type(features), pd.DataFrame)
