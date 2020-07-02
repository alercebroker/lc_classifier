import os
import unittest
import numpy as np
import pandas as pd
from late_classifier.features import PeriodExtractor
from late_classifier.features import DetectionsPreprocessorZTF


FILE_PATH = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_PATH = os.path.abspath(os.path.join(FILE_PATH, "../data"))


class TestPeriodExtractor(unittest.TestCase):
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
        self.detections = DetectionsPreprocessorZTF().get_magpsf_ml(
            self.detections,
            objects=self.fake_objects
        )

    def test_periods(self):
        period_extractor = PeriodExtractor()
        periods = period_extractor.compute_features(
            detections=self.detections)
        periods['catalog_period'] = self.labels.loc[periods.index].period
        print(periods)

    def test_shared_data(self):
        shared_data = dict()
        period_extractor = PeriodExtractor()
        _ = period_extractor.compute_features(
            detections=self.detections,
            shared_data=shared_data
        )
        self.assertTrue(len(shared_data.keys()) > 0)


if __name__ == '__main__':
    unittest.main()
