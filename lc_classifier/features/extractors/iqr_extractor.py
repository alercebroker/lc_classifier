from typing import List

from ..core.base import FeatureExtractorSingleBand
import scipy.stats as sstats
import pandas as pd
import logging


class IQRExtractor(FeatureExtractorSingleBand):
    def get_features_keys(self) -> List[str]:
        return ['iqr']

    def get_required_keys(self) -> List[str]:
        return ['fid', 'magpsf_ml']

    def compute_feature_in_one_band(self, detections, band=None, **kwargs):
        oids = detections.index.unique()
        iqrs = []
        columns = self.get_features_keys_with_band(band)
        for oid in oids:
            oid_detections = detections.loc[[oid]]
            if band not in oid_detections.fid.values:
                logging.debug(
                    f'extractor=IQR  object={oid}  required_cols={self.get_required_keys()}  band={band}')
                nan_df = self.nan_df(oid)
                nan_df.columns = columns
                iqrs.append(nan_df)
                continue

            oid_band_detections = oid_detections[oid_detections.fid == band]
            mag_dets = oid_band_detections["magpsf_ml"]
            iqr = sstats.iqr(mag_dets.values)
            iqr_df = pd.DataFrame([iqr], columns=columns, index=[oid])
            iqrs.append(iqr_df)
        iqrs = pd.concat(iqrs, axis=0)
        iqrs.index.name = 'oid'
        return iqrs
