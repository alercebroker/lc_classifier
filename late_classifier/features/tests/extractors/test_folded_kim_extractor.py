import os
import io
import unittest
import numpy as np
import pandas as pd
from late_classifier.features import FoldedKimExtractor
from late_classifier.features import PeriodExtractor
from late_classifier.features import DetectionsPreprocessorZTF
import logging


FILE_PATH = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_PATH = os.path.abspath(os.path.join(FILE_PATH, "../data"))


class FoldedKimExtractorTest(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(0)
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
            objects=self.fake_objects)
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

    def test_extractor_without_periodogram(self):
        output = io.StringIO()
        stream_handler = logging.StreamHandler(output)
        self.logger.addHandler(stream_handler)
        extractor = FoldedKimExtractor()
        features = extractor.compute_features(detections=self.detections)
        self.assertTrue(
            'Folded Kim extractor was not provided with period '
            'data, so a periodogram is being computed' in output.getvalue())
        n_bands = len(self.detections.fid.unique())
        self.assertEqual(
            features.shape,
            (len(self.fake_objects),
             n_bands * len(extractor.get_features_keys())))

    def test_extractor(self):
        output = io.StringIO()
        stream_handler = logging.StreamHandler(output)
        self.logger.addHandler(stream_handler)
        period_extractor = PeriodExtractor()
        shared_data = dict()
        _ = period_extractor.compute_features(
            detections=self.detections,
            shared_data=shared_data)
        extractor = FoldedKimExtractor()
        features = extractor.compute_features(
            detections=self.detections,
            shared_data=shared_data
        )
        self.assertTrue(
            'Folded Kim extractor was not provided with period '
            'data, so a periodogram is being computed' not in output.getvalue())
        n_bands = len(self.detections.fid.unique())
        self.assertEqual(
            features.shape,
            (len(self.fake_objects),
             n_bands * len(extractor.get_features_keys())))


if __name__ == '__main__':
    unittest.main()
