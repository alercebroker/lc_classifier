import pandas as pd


class FeatureExtractorDecorator:
    def __init__(self, feature_extractor):
        self.decorated_feature_extractor = feature_extractor

    def compute_features(self, detections):
        raise NotImplementedError('FeatureExtractorDecorator is an interface')


class FeatureExtractorPerBand(FeatureExtractorDecorator):
    def __init__(self, feature_extractor, bands):
        super().__init__(feature_extractor)
        self.bands = bands

    def compute_features(self, detections):
        features_df_list = []
        for band in self.bands:
            band_detections = detections[detections.fid == band]
            band_features = self.decorated_feature_extractor.compute_features(band_detections)
            band_features = band_features.rename(
                lambda x: f'{x}_{band}',
                axis='columns')
            features_df_list.append(band_features)
        features = pd.concat(features_df_list, axis=1, join='outer', sort=True)
        return features


class FeatureExtractorFromSingleLCExtractor(FeatureExtractorDecorator):
    def __init__(self, feature_extractor):
        super().__init__(feature_extractor)

    def compute_features(self, detections):
        oids = detections.index.unique()
        features = []
        for oid in oids:
            light_curve = detections.loc[[oid]]
            lc_features = self.decorated_feature_extractor.compute_features(light_curve)
            features.append(lc_features)
        features = pd.concat(features)
        return features


class FeatureExtractorPerBandFromSingleLC(FeatureExtractorPerBand):
    def __init__(self, feature_extractor, bands=[1, 2]):
        super().__init__(feature_extractor, bands)

    def compute_features(self, detections):
        oids = detections.index.unique()
        features = []
        for oid in oids:
            for band in self.bands:
                det = detections.query(f'fid=={band} and oid=="{oid}"')
                fs = self.decorated_feature_extractor.compute_features(det)
                features.append(fs)
        features = pd.concat(features)
        return features


class FeatureComputer:
    def compute(self, df):
        df_by_oid = df.groupby(level=0)

