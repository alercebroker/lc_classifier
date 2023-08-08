from typing import Tuple
import numpy as np
import pandas as pd
from ..core.base import FeatureExtractor


class TimespanExtractor(FeatureExtractor):
    def get_features_keys(self) -> Tuple[str, ...]:
        return ('Timespan',)
    
    def get_required_keys(self) -> Tuple[str, ...]:
        return ('time', 'difference_flux', 'difference_flux_error')

    def _compute_features(self, detections, **kwargs):
        return self._compute_features_from_df_groupby(
            detections.groupby(level=0),
            **kwargs)

    def _compute_features_from_df_groupby(
            self, detections, **kwargs) -> pd.DataFrame:
        columns = self.get_features_keys()
        
        def aux_function(oid_detections, **kwargs):
            fluxes = oid_detections['difference_flux'].values
            errors = oid_detections['difference_flux_error'].values
            times = oid_detections['time'].values 

            detected = np.abs(fluxes) > 5*errors
            detected_times = times[detected]
            if len(detected_times) == 0:
                timespan = 0.0
            else:
                timespan = np.max(detected_times) - np.min(detected_times)
            out = pd.Series(
                data=[timespan],
                index=columns
            )
            return out

        timespan_df = detections.apply(aux_function)
        timespan_df.index.name = 'oid'
        return timespan_df