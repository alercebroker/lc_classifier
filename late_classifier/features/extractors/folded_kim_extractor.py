from typing import List
import logging
import numpy as np
import pandas as pd

from ..core.base import FeatureExtractorSingleBand
from ..extractors import PeriodExtractor


class FoldedKimExtractor(FeatureExtractorSingleBand):
    """https://github.com/dwkim78/upsilon/blob/
    64b2b7f6e3b3991a281182549959aabd385a6f28/
    upsilon/extract_features/extract_features.py#L151"""
    def get_features_keys(self) -> List[str]:
        return [
            'Psi_CS',
            'Psi_eta'
        ]

    def get_required_keys(self) -> List[str]:
        return [
            'mjd',
            'magpsf_ml'
        ]

    def compute_feature_in_one_band(self, detections: pd.DataFrame, band=None, **kwargs) -> pd.DataFrame:
        if ('shared_data' in kwargs.keys() and
                'period' in kwargs['shared_data'].keys()):
            periods = kwargs['shared_data']['period']
        else:
            logging.info('Folded Kim extractor was not provided with period '
                         'data, so a periodogram is being computed')
            period_extractor = PeriodExtractor()
            periods = period_extractor.compute_features(detections)
        oids = detections.index.unique()
        features = []
        for oid in oids:
            oid_detections = detections.loc[[oid]]
            if band not in oid_detections.fid.values:
                logging.info(
                    f'extractor=Folded Kim extractor object={oid} '
                    f'required_cols={self.get_required_keys()}  band={band}')
                features.append([np.nan] * len(self.get_features_keys()))
                continue

            oid_band_detections = oid_detections[oid_detections.fid == band]
            time = oid_band_detections['mjd'].values
            oid_period = periods[['Multiband_period']].loc[[oid]].values.flatten()
            folded_time = np.mod(time, 2 * oid_period) / (2 * oid_period)
            magnitude = oid_band_detections['magpsf_ml'].values
            sorted_mags = magnitude[np.argsort(folded_time)]
            sigma = np.std(sorted_mags)
            m = np.mean(sorted_mags)
            lc_len = len(sorted_mags)
            s = np.cumsum(sorted_mags - m) * 1.0 / (lc_len * sigma)
            psi_cumsum = np.max(s) - np.min(s)
            sigma_squared = sigma ** 2
            psi_eta = (1.0 / ((lc_len - 1) * sigma_squared) *
                       np.sum(np.power(sorted_mags[1:] - sorted_mags[:-1], 2)))
            features.append([psi_cumsum, psi_eta])
        features = pd.DataFrame(
            data=np.array(features),
            columns=self.get_features_keys_with_band(band),
            index=oids)
        return features
