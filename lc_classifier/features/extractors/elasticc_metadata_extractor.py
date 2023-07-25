import pandas as pd
from typing import Tuple
from functools import lru_cache
from ..core.base import FeatureExtractor


class ElasticcMetadataExtractor(FeatureExtractor):
    @lru_cache(1)
    def get_features_keys(self) -> Tuple[str, ...]:
        return 'redshift_helio', 'mwebv'

    @lru_cache(1)
    def get_required_keys(self) -> Tuple[str, ...]:
        # not very useful
        return tuple()

    def _compute_features(self, detections, **kwargs):
        return self._compute_features_from_df_groupby(
            detections.groupby(level=0),
            **kwargs)

    def _compute_features_from_df_groupby(
            self, detections, **kwargs) -> pd.DataFrame:
        columns = self.get_features_keys()
        metadata = kwargs['metadata']

        def aux_function(oid_detections, **kwargs):
            oid = oid_detections.index.values[0]
            metadata_lightcurve = metadata.loc[oid]
            redshift_helio = metadata_lightcurve['REDSHIFT_HELIO']
            mwebv = metadata_lightcurve['MWEBV']

            out = pd.Series(
                data=[redshift_helio, mwebv],
                index=columns)
            return out

        features = detections.apply(aux_function)
        features.index.name = 'oid'
        return features
    

class ElasticcFullMetadataExtractor(FeatureExtractor):
    @lru_cache(1)
    def get_features_keys(self) -> Tuple[str, ...]:
        columns = [
            'MWEBV', 'MWEBV_ERR', 'REDSHIFT_HELIO', 'REDSHIFT_HELIO_ERR',       
            'HOSTGAL2_ELLIPTICITY', 'HOSTGAL2_MAG_Y', 'HOSTGAL2_MAG_g',
            'HOSTGAL2_MAG_i', 'HOSTGAL2_MAG_r', 'HOSTGAL2_MAG_u', 'HOSTGAL2_MAG_z',
            'HOSTGAL2_MAGERR_Y', 'HOSTGAL2_MAGERR_g', 'HOSTGAL2_MAGERR_i',
            'HOSTGAL2_MAGERR_r', 'HOSTGAL2_MAGERR_u', 'HOSTGAL2_MAGERR_z',
            'HOSTGAL2_SNSEP', 'HOSTGAL2_SQRADIUS', 'HOSTGAL2_PHOTOZ',
            'HOSTGAL2_PHOTOZ_ERR', 'HOSTGAL2_ZPHOT_Q000', 'HOSTGAL2_ZPHOT_Q010',
            'HOSTGAL2_ZPHOT_Q020', 'HOSTGAL2_ZPHOT_Q030', 'HOSTGAL2_ZPHOT_Q040',
            'HOSTGAL2_ZPHOT_Q050', 'HOSTGAL2_ZPHOT_Q060', 'HOSTGAL2_ZPHOT_Q070',
            'HOSTGAL2_ZPHOT_Q080', 'HOSTGAL2_ZPHOT_Q090', 'HOSTGAL2_ZPHOT_Q100',
            'HOSTGAL2_SPECZ', 'HOSTGAL2_SPECZ_ERR', 'HOSTGAL_ELLIPTICITY',
            'HOSTGAL_MAG_Y', 'HOSTGAL_MAG_g', 'HOSTGAL_MAG_i', 'HOSTGAL_MAG_r',
            'HOSTGAL_MAG_u', 'HOSTGAL_MAG_z', 'HOSTGAL_MAGERR_Y',
            'HOSTGAL_MAGERR_g', 'HOSTGAL_MAGERR_i', 'HOSTGAL_MAGERR_r',
            'HOSTGAL_MAGERR_u', 'HOSTGAL_MAGERR_z', 'HOSTGAL_SNSEP',
            'HOSTGAL_SQRADIUS', 'HOSTGAL_PHOTOZ', 'HOSTGAL_PHOTOZ_ERR',
            'HOSTGAL_ZPHOT_Q000', 'HOSTGAL_ZPHOT_Q010', 'HOSTGAL_ZPHOT_Q020',
            'HOSTGAL_ZPHOT_Q030', 'HOSTGAL_ZPHOT_Q040', 'HOSTGAL_ZPHOT_Q050',
            'HOSTGAL_ZPHOT_Q060', 'HOSTGAL_ZPHOT_Q070', 'HOSTGAL_ZPHOT_Q080',
            'HOSTGAL_ZPHOT_Q090', 'HOSTGAL_ZPHOT_Q100', 'HOSTGAL_SPECZ',
            'HOSTGAL_SPECZ_ERR']
        return columns

    @lru_cache(1)
    def get_required_keys(self) -> Tuple[str, ...]:
        # not very useful
        return tuple()

    def _compute_features(self, detections, **kwargs):
        return self._compute_features_from_df_groupby(
            detections.groupby(level=0),
            **kwargs)

    def _compute_features_from_df_groupby(
            self, detections, **kwargs) -> pd.DataFrame:
        columns = self.get_features_keys()
        metadata = kwargs['metadata']

        force_snids = None
        if 'force_snids' in kwargs.keys():
            force_snids = kwargs['force_snids']

        def aux_function(oid_detections, **kwargs):
            oid = oid_detections.index.values[0]
            metadata_lightcurve = metadata.loc[oid]

            out = pd.Series(
                data=metadata_lightcurve[columns],
                index=columns)
            return out
        if force_snids is None:
            features = detections.apply(aux_function)
        else:
            features = metadata[columns].loc[force_snids]
        features.index.name = 'oid'
        return features
