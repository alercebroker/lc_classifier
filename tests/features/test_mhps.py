import unittest
import numpy as np
import pandas as pd

from lc_classifier.features.preprocess import ElasticcPreprocessor
from lc_classifier.features.extractors.mhps_extractor import MHPSFluxExtractor
import os


class TestMHPSElasticc(unittest.TestCase):
    def setUp(self) -> None:
        this_dir = os.path.dirname(os.path.abspath(__file__))
        example_lc_filename = os.path.join(
            this_dir,
            "../../data_examples/lc_sample_wo_nev_comp.parquet")
        self.light_curve_df = pd.read_parquet(example_lc_filename)
        preprocessor = ElasticcPreprocessor()
        self.light_curve_df = preprocessor.preprocess(self.light_curve_df)

    def test_fit(self):
        extractor = MHPSFluxExtractor([b for b in 'ugrizY'])
        features = extractor.compute_features(self.light_curve_df)
        # print(features.isna().mean(axis=0))


if __name__ == '__main__':
    unittest.main()
