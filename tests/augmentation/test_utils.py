import unittest
import pandas as pd

from lc_classifier.augmentation import utils


class TestUtils(unittest.TestCase):
    def setUp(self) -> None:
        self.train_features = pd.read_csv('data_examples/2000_features.csv', index_col="objectId")
        self.train_labels = pd.read_csv('data_examples/2000_labels.csv', index_col="objectId")

    def test_utils_function(self):
        new_features, new_labels = utils.augment_just_training_set_features(
            self.train_features.iloc[:1500],
            self.train_labels[:1500],
            self.train_features[1500:],
            self.train_labels[1500:])
        self.assertEqual(len(new_features), 1500)
