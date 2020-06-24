from typing import List

from ..core.base import FeatureExtractorSingleBand
import pandas as pd
import logging


class SupernovaeDetectionFeatureExtractor(FeatureExtractorSingleBand):
    def get_features_keys(self) -> List[str]:
        return ['delta_mag_fid',
                'delta_mjd_fid',
                'first_mag',
                'mean_mag',
                'min_mag',
                'n_det',
                'n_neg',
                'n_pos',
                'positive_fraction']

    def get_required_keys(self) -> List[str]:
        return ["isdiffpos", "magpsf_corr", "magpsf", "mjd"]

    def compute_feature_in_one_band(self, detections, band=None, **kwargs):
        """

        Parameters
        ----------
        detections :class:pandas.`DataFrame`
        DataFrame with single band detections of an object.

        band :class:int

        kwargs:
        objects : class:pandas.`DataFrame`
        Dataframe with the objects table.

        Returns :class:pandas.`DataFrame`
        -------

        """

        required = ['objects']
        for key in required:
            if key not in kwargs:
                raise Exception(f'SupernovaeDetectionFeatureExtractor requires {key} argument')

        objects = kwargs['objects']

        oids = detections.index.unique()
        sn_det_results = []

        detections = detections.sort_values('mjd')
        columns = self.get_features_keys_with_band(band)

        for oid in oids:
            oid_detections = detections.loc[[oid]]
            oid_objects = objects.loc[[oid]]

            if band not in oid_detections.fid.values:
                logging.info(
                    f'extractor=SN detection object={oid} required_cols={self.get_required_keys()} band={band}')
                nan_df = self.nan_df(oid)
                nan_df.columns = columns
                sn_det_results.append(nan_df)
                continue

            oid_band_detections = oid_detections[oid_detections.fid == band]

            objects_corrected = oid_objects.corrected.values[0]

            if objects_corrected:

                n_pos = len(oid_band_detections[oid_band_detections.isdiffpos > 0])
                n_neg = len(oid_band_detections[oid_band_detections.isdiffpos < 0])
                min_mag = oid_band_detections['magpsf_corr'].values.min()
                first_mag = oid_band_detections['magpsf_corr'].values[0]
                delta_mjd_fid = oid_band_detections['mjd'].values[-1] - oid_band_detections['mjd'].values[0]
                delta_mag_fid = oid_band_detections['magpsf_corr'].values.max() - min_mag
                positive_fraction = n_pos/(n_pos + n_neg)
                mean_mag = oid_band_detections['magpsf_corr'].values.mean()

            else:

                n_pos = len(oid_band_detections[oid_band_detections.isdiffpos > 0])
                n_neg = len(oid_band_detections[oid_band_detections.isdiffpos < 0])
                min_mag = oid_band_detections['magpsf'].values.min()
                first_mag = oid_band_detections['magpsf'].values[0]
                delta_mjd_fid = oid_band_detections['mjd'].values[-1] - oid_band_detections['mjd'].values[0]
                delta_mag_fid = oid_band_detections['magpsf'].values.max() - min_mag
                positive_fraction = n_pos/(n_pos + n_neg)
                mean_mag = oid_band_detections['magpsf'].values.mean()

            data = [delta_mag_fid,
                    delta_mjd_fid,
                    first_mag,
                    mean_mag,
                    min_mag,
                    n_neg + n_pos,
                    n_neg,
                    n_pos,
                    positive_fraction]
            sn_det_df = pd.DataFrame.from_records([data], columns=columns, index=[oid])
            sn_det_results.append(sn_det_df)
        sn_det_results = pd.concat(sn_det_results, axis=0)
        sn_det_results.index.name = 'oid'
        return sn_det_results
