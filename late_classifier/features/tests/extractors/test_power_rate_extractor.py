import unittest
import os
import numpy as np
import pandas as pd
from late_classifier.features import PowerRateExtractor
from late_classifier.features import PeriodExtractor
import logging
import io


FILE_PATH = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_PATH = os.path.abspath(os.path.join(FILE_PATH, "../data"))


class TestPowerRateExtractor(unittest.TestCase):
    def setUp(self) -> None:
        self.detections = pd.read_pickle(
            os.path.join(EXAMPLES_PATH, 'periodic_light_curves.pkl'))
        self.labels = pd.read_pickle(
            os.path.join(EXAMPLES_PATH, 'periodic_light_curve_labels.pkl'))
        self.detections['sigmapsf_corr_ext'] = self.detections['sigmapsf_corr']
        self.fake_objects = pd.DataFrame(
            index=self.labels.index,
            columns=['corrected'],
            data=np.random.choice(a=[False, True], size=(len(self.labels),)))
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

    def test_power_rate_without_periodogram(self):
        output = io.StringIO()
        stream_handler = logging.StreamHandler(output)
        self.logger.addHandler(stream_handler)
        extractor = PowerRateExtractor()
        power_rates = extractor.compute_features(
            self.detections,
            objects=self.fake_objects)
        self.assertTrue(
            'PowerRateExtractor was not provided with periodogram '
            'data, so a periodogram is being computed' in output.getvalue())
        self.assertEqual(
            power_rates.shape,
            (len(self.fake_objects), len(extractor.get_features_keys())))

    def test_power_rate(self):
        output = io.StringIO()
        stream_handler = logging.StreamHandler(output)
        self.logger.addHandler(stream_handler)
        period_extractor = PeriodExtractor()
        shared_data = dict()
        _ = period_extractor.compute_features(
            detections=self.detections,
            objects=self.fake_objects,
            shared_data=shared_data)
        extractor = PowerRateExtractor()
        power_rates = extractor.compute_features(
            self.detections,
            objects=self.fake_objects,
            shared_data=shared_data)
        self.assertTrue(
            'PowerRateExtractor was not provided with periodogram '
            'data, so a periodogram is being computed' not in output.getvalue())
        self.assertEqual(
            power_rates.shape,
            (len(self.fake_objects), len(extractor.get_features_keys())))


if __name__ == '__main__':
    unittest.main()
