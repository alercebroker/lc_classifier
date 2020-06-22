import unittest
import numpy as np
import pandas as pd
from late_classifier.classifier.metrics import kaggle_score


class TestKaggleScore(unittest.TestCase):
    def test_equal_probs(self):
        label_values = np.random.randint(0, 5, size=[100, 1])
        labels = pd.DataFrame(
            data=label_values,
            columns=['classALeRCE']
        )

        prob_values = np.ones((100, 5), dtype=np.float) / 5.0
        probs = pd.DataFrame(
            data=prob_values,
            columns=range(5)
        )
        kaggle_score_value = kaggle_score(probs, labels)
        self.assertLess(np.abs(kaggle_score_value - np.log(5)), 1e-6)

    def test_perfect_score(self):
        label_values = np.random.randint(0, 5, size=[100, 1])
        labels = pd.DataFrame(
            data=label_values,
            columns=['classALeRCE']
        )

        prob_values = np.zeros((100, 5), dtype=np.float)
        for row in range(len(labels)):
            prob_values[row, label_values[row]] = 1.0

        probs = pd.DataFrame(
            data=prob_values,
            columns=range(5)
        )
        kaggle_score_value = kaggle_score(probs, labels)
        self.assertLess(np.abs(kaggle_score_value - 0.0), 1e-6)


if __name__ == '__main__':
    unittest.main()
