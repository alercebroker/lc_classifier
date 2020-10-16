import unittest
import numpy as np
import pandas as pd

from lc_classifier.augmentation.simple_augmentation import ShortTransientAugmenter


class TestShortTransientAugmenter(unittest.TestCase):
    def setUp(self) -> None:
        self.detections = pd.read_csv('data_examples/30_sne_detections_corr.csv', index_col="objectId")
        self.non_detections = pd.read_csv('data_examples/30_sne_non_detections.csv', index_col="objectId")
        self.non_detections["mjd"] = self.non_detections['jd'] - 2400000.5
        self.labels = pd.read_csv('data_examples/30_sne_labels.csv', index_col="objectId")

    def test_short_transient_augmenter(self):
        augmenter = ShortTransientAugmenter()
        new_detections, new_non_detections, new_labels = augmenter.augment(
            self.detections, self.non_detections, self.labels, multiplier=5
        )

        self.assertGreater(len(new_detections), len(self.detections))

        self.assertLess(
            np.mean(new_detections.groupby(level=0).count()['mjd'].values),
            np.mean(self.detections.groupby(level=0).count()['mjd'].values)
        )
