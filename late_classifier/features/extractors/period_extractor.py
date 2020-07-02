from typing import List
import numpy as np
import pandas as pd
from ..core.base import FeatureExtractor
from P4J import MultiBandPeriodogram


class PeriodExtractor(FeatureExtractor):
    def __init__(self):
        self.periodogram_computer = MultiBandPeriodogram(method='MHAOV')

    def get_features_keys(self) -> List[str]:
        return ['Multiband_period', 'Period_fit']

    def get_required_keys(self) -> List[str]:
        return [
            'mjd',
            'magpsf',
            'magpsf_corr',
            'fid',
            'sigmapsf_corr_ext',
            'sigmapsf']

    def _compute_features(self, detections: pd.DataFrame, **kwargs) -> pd.DataFrame:
        required = ['objects']
        for key in required:
            if key not in kwargs:
                raise Exception(f'Period extractor requires {key} argument')

        objects = kwargs['objects']

        oids = detections.index.unique()
        features = []

        detections = detections.sort_values('mjd')

        periodograms = {}
        for oid in oids:
            oid_detections = detections.loc[[oid]]
            oid_objects = objects.loc[[oid]]

            oid_detections = oid_detections.groupby('fid').filter(
                lambda x: len(x) > 5)

            objects_corrected = oid_objects.corrected.values[0]
            if objects_corrected:
                data_column_names = ['magpsf_corr', 'mjd', 'sigmapsf_corr_ext']
            else:
                data_column_names = ['magpsf', 'mjd', 'sigmapsf']

            self.periodogram_computer.set_data(
                mjds=oid_detections[[data_column_names[1]]].values,
                mags=oid_detections[[data_column_names[0]]].values,
                errs=oid_detections[[data_column_names[2]]].values,
                fids=oid_detections[['fid']].values)

            self.periodogram_computer.frequency_grid_evaluation(
                fmin=1e-3, fmax=20.0, fresolution=1e-3)
            self.frequencies = self.periodogram_computer.finetune_best_frequencies(
                n_local_optima=10, fresolution=1e-4)
            best_freq, best_per = self.periodogram_computer.get_best_frequencies()

            freq, per = self.periodogram_computer.get_periodogram()
            period_candidate = 1.0 / best_freq[0]

            # Significance estimation
            entropy_best_n = 100
            top_values = np.sort(per)[-entropy_best_n:]
            normalized_top_values = top_values + 1e-2
            normalized_top_values = normalized_top_values / np.sum(normalized_top_values)
            entropy = (-normalized_top_values * np.log(normalized_top_values)).sum()
            significance = 1 - entropy / np.log(entropy_best_n)

            object_features = pd.DataFrame(
                data=[[period_candidate, significance]],
                columns=self.get_features_keys(),
                index=[oid]
            )
            features.append(object_features)
            periodograms[oid] = {
                'freq': freq,
                'per': per
            }

        features = pd.concat(features, axis=0, sort=True)
        features.index.name = 'oid'
        if 'shared_data' in kwargs.keys():
            kwargs['shared_data']['period'] = features
            kwargs['shared_data']['periodogram'] = periodograms
        return features
