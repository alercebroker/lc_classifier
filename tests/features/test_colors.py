import unittest
import os
import pandas as pd

from lc_classifier.features.preprocess import ElasticcPreprocessor
from lc_classifier.features import ElasticcColorFeatureExtractor


class TestElasticcColors(unittest.TestCase):
    def setUp(self) -> None:
        self.this_dir = os.path.dirname(os.path.abspath(__file__))
        self.preprocessor = ElasticcPreprocessor()
        bands = [b for b in 'ugrizY']
        self.color_extractor = ElasticcColorFeatureExtractor(bands)
        
    def test_colors_nice(self):
        nice_lc_filename = os.path.join(
            self.this_dir,
            "../../data_examples/elasticc_lc_chunk.parquet"
        )
        nice_lc_df = pd.read_parquet(nice_lc_filename)
        nice_lc_df = self.preprocessor.preprocess(nice_lc_df)
        features = self.color_extractor.compute_features(nice_lc_df)

    def test_colors_hard(self):
        hard_lc_filename = os.path.join(
            self.this_dir,
            "../../data_examples/lc_sample_wo_nev_comp.parquet"
        )
        hard_lc_df = pd.read_parquet(hard_lc_filename)
        hard_lc_df = self.preprocessor.preprocess(hard_lc_df)
        features = self.color_extractor.compute_features(hard_lc_df)


if __name__ == '__main__':
    unittest.main()
