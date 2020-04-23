from late_classifier.features.extractors.sn_parametric_model_computer import SNParametricModelExtractor

from late_classifier.features.core.base import FeatureExtractor
from functools import reduce
import pandas as pd
import warnings


# TODO deprecate this, it's not being used
class JustSNModelFeaturesComputer(FeatureExtractor):
    def __init__(self, bands):
        super().__init__()
        self.bands = bands
        self.sn_model_extractor = SNParametricModelExtractor()
        
        self.features_keys = self.get_features_keys()


    def get_features_keys(self):
        return self.sn_model_extractor.features_keys

    def enough_alerts(self, object_alerts):
        """
        Verify if an object has enough alerts.
        Parameters
        ----------
        object_alerts: :class:pandas. `DataFrame`
        DataFrame with detections of only one object.

        Returns Boolean
        -------

        """
        if len(object_alerts) <= 0 and type(object_alerts) is not pd.DataFrame and len(object_alerts.index.unique()) == 1:
            return False
        booleans = list(map(lambda band: len(object_alerts.fid == band) > 5, self.bands))
        return reduce(lambda x, y: x | y, booleans)

    def single_band_features(self, detections, non_detections):
        """
        Compute a single band features of detections and non detections. Given a DataFrame of
        different objects this extract single band features for each objects.

        Parameters
        ----------
        detections: :class:pandas.`DataFrame`
        DataFrame that contains detections of different astronomical objects.

        non_detections: :class:pandas.`DataFrame`
        DataFrame that contains non detections of different an astronomical objects.

        Returns DataFrame with single band features joined
        -------

        """
        oids = detections.index.unique()
        features = []

        for oid in oids:
            try:
                object_alerts = detections.loc[[oid]]
                if self.enough_alerts(object_alerts):
                    try:
                        object_non_detections = non_detections.loc[[oid]]
                    except KeyError:
                        object_non_detections = pd.DataFrame(columns=["fid", "mjd"])

                    for band in self.bands:
                        n_samples = len(object_alerts[object_alerts.fid == band])
                        enough_alerts_band = n_samples > 5
                        data_detections = object_alerts[object_alerts.fid == band]
                        data_non_detections = object_non_detections[
                            object_non_detections.fid == band]
                        if enough_alerts_band:
                            try:
                                sn_model_params = self.sn_model_extractor.compute_features(data_detections)
                                df = sn_model_params
                                df['fid'] = band
                                
                                # features = pd.concat([features, df], sort=True)
                                features.append(df)
                            except Exception as e:
                                warnings.warn('Exception when computing features of '+str(oid)+str(e))
                        # else:
                        #     df = pd.DataFrame([[band, oid]], columns=['fid', 'oid'])
                        #     df = df.set_index('oid')
                        #     features = features.append(df, sort=False)
                    
            except Exception as e:
                #print(e)
                raise Exception('Fatal error 0x187AF')
        return pd.concat(features)

    def compute_features(self, detections, **kwargs):
        """

        Parameters
        ----------
        detections :class:pandas.`DataFrame`
        kwargs Possible non_detections :class:pandas.`DataFrame`

        Returns DataFrame with all features
        -------

        """
        required = ['non_detections']
        for key in required:
            if key not in kwargs:
                raise Exception(f'HierarchicalFeaturesComputer requires {key} argument')
        if type(detections) is not pd.DataFrame:
            raise Exception(
                f'HierarchicalFeaturesComputer requires detections in pd.Dataframe datatype, not {type(detections)}')

        detections = detections.sort_values('mjd')
        non_detections = kwargs['non_detections'].sort_values('mjd')

        single = self.single_band_features(detections, non_detections)

        single_band_features = [f for f in single.columns.values if f != 'fid']

        if len(single) > 0:
            fids = single.fid.unique()
            single_band_dfs = []
            for fid in fids:
                features_band_fid = single[single.fid == fid]
                features_band_fid = features_band_fid[single_band_features]
                features_band_fid = features_band_fid.rename(
                    lambda x: x + f'_{fid}',
                    axis='columns')
                single_band_dfs.append(features_band_fid)
            single = pd.concat(single_band_dfs, axis=1, join='outer', sort=True)
            df = single
            return df
        else:
            return pd.DataFrame([])
