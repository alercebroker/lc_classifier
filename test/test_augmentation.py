import unittest
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

from late_classifier.augmentation.simple_augmentation import ShortTransientAugmenter


class TestShortTransientAugmenter(unittest.TestCase):
    def setUp(self) -> None:
        self.detections = pd.read_pickle('data/detections.pkl')
        self.non_detections = pd.read_pickle('data/non_detections.pkl')
        self.labels = pd.read_pickle('data/labels.pkl')

    def test_(self):
        augmenter = ShortTransientAugmenter()
        new_detections, new_non_detections, new_labels = augmenter.augment(
            self.detections, self.non_detections, self.labels, multiplier=5
        )

        # print(self.detections.shape, new_detections.shape)
        self.assertGreater(len(new_detections), len(self.detections))

        # plt.subplot(2, 1, 1)
        # plt.hist(self.detections.groupby(level=0).count()['mjd'].values, bins=40)
        # plt.subplot(2, 1, 2)
        # plt.hist(new_detections.groupby(level=0).count()['mjd'].values, bins=40)
        # plt.show()

        self.assertLess(
            np.mean(new_detections.groupby(level=0).count()['mjd'].values),
            np.mean(self.detections.groupby(level=0).count()['mjd'].values)
        )
