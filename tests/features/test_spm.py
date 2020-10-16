import unittest
import numpy as np
from lc_classifier.features.extractors.sn_model_tf import SNModel


class TestSPM(unittest.TestCase):
    def setUp(self) -> None:
        self.sn_model = SNModel()

    def test_run(self):
        times = np.linspace(-50, 200, 400)
        flux = self.sn_model(times)
        flux2 = flux * 1.2
        times2 = times * 1.6
        loss_value = self.sn_model.fit(times2, flux2)
        prediction = self.sn_model(times)
        self.assertEqual(len(times), len(prediction))
        print(loss_value)
