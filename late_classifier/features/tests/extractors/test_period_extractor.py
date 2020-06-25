import os
import unittest
import numpy as np
import pandas as pd
from late_classifier.features import PeriodExtractor


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

    def test_periods(self):
        period_extractor = PeriodExtractor()
        periods = period_extractor.compute_features(
            detections=self.detections, objects=self.fake_objects)
        periods['catalog_period'] = self.labels.loc[periods.index].period
        print(periods)


if __name__ == '__main__':
    unittest.main()
