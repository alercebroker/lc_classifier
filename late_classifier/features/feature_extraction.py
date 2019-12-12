import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from turbofats import NewFeatureSpace
from late_classifier.features.preprocessing import DetectionsPreprocessor


class FeatureExtractor:
    def compute_features(self, detections):
        raise NotImplementedError('FeatureExtractor is an interface')


class SingleBandFeatureExtractor(FeatureExtractor):
    def compute_features(self, detections):
        n_bands = len(detections.fid.unique())
        if n_bands > 1:
            raise Exception('SingleBandFeatureExtractor cannot handle multiple bands')
        return self._compute_features(detections)

    def _compute_features(self, detections):
        raise NotImplementedError('SingleBandFeatureExtractor is an abstract class')


class FeatureExtractorDecorator:
    def __init__(self, feature_extractor):
        self.decorated_feature_extractor = feature_extractor

    def compute_features(self, detections):
        raise NotImplementedError('FeatureExtractorDecorator is an interface')


class FeatureExtractorPerBand(FeatureExtractorDecorator):
    def __init__(self, feature_extractor):
        super().__init__(feature_extractor)
        self.bands = [1, 2]

    def compute_features(self, detections):
        features_df_list = []
        for band in self.bands:
            band_detections = detections[detections.fid == band]
            band_features = self.decorated_feature_extractor.compute_features(band_detections)
            band_features = band_features.rename(
                lambda x: x + f'_{band}',
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


class FeatureExtractorFromFEList(FeatureExtractor):
    def __init__(self, feature_extractor_list):
        self.feature_extractor_list = feature_extractor_list

    def compute_features(self, detections):
        all_features = []
        for feature_extractor in self.feature_extractor_list:
            features = feature_extractor.compute_features(detections)
            all_features.append(features)
        all_features = pd.concat(all_features, axis=1, join='outer', sort=True)
        return all_features


class TurboFatsFeatureExtractor(SingleBandFeatureExtractor):
    def __init__(self):
        self.features_keys = [
            'Amplitude', 'AndersonDarling', 'Autocor_length',
            'Beyond1Std',
            'Con', 'Eta_e',
            'Gskew',
            'MaxSlope', 'Mean', 'Meanvariance', 'MedianAbsDev',
            'MedianBRP', 'PairSlopeTrend', 'PercentAmplitude', 'Q31',
            'PeriodLS_v2',
            'Period_fit_v2', 'Psi_CS_v2', 'Psi_eta_v2', 'Rcs',
            'Skew', 'SmallKurtosis', 'Std',
            'StetsonK', 'Harmonics',
            'Pvar', 'ExcessVar',
            'GP_DRW_sigma', 'GP_DRW_tau', 'SF_ML_amplitude', 'SF_ML_gamma',
            'IAR_phi', 'LinearTrend'
        ]
        self.feature_space = NewFeatureSpace(self.features_keys)

    def _compute_features(self, detections):
        return self.feature_space.calculate_features(detections)


class GalacticCoordinatesComputer(FeatureExtractor):
    def compute_features(self, detections):
        radec_df = detections[['ra', 'dec']].groupby(level=0).mean()
        coordinates = SkyCoord(
            ra=radec_df['ra'], dec=radec_df['dec'], frame='icrs', unit='deg')
        galactic = coordinates.galactic
        np_galactic = np.stack((galactic.b.degree, galactic.l.degree), axis=-1)
        df = pd.DataFrame(np_galactic, index=radec_df.index, columns=['gal_b', 'gal_l'])
        df.index.name = 'oid'
        return df


class SGScoreComputer(FeatureExtractor):
    def compute_features(self, detections):
        sgscore = detections['sgscore1'].groupby(level=0).median()
        return sgscore


class ColorFeatureComputer(FeatureExtractor):
    def compute_features(self, detections):
        g_band_mag = detections[detections.fid == 1]['magpsf_corr']
        r_band_mag = detections[detections.fid == 2]['magpsf_corr']
        g_r_max = g_band_mag.groupby(level=0).min() - r_band_mag.groupby(level=0).min()
        g_r_max.rename('g-r_max', inplace=True)

        g_r_mean = g_band_mag.groupby(level=0).mean() - r_band_mag.groupby(level=0).mean()
        g_r_mean.rename('g-r_mean', inplace=True)
        features = pd.concat([g_r_max, g_r_mean], axis=1)
        return features


class RealBogusComputer(FeatureExtractor):
    def compute_features(self, detections):
        rb = detections['rb'].groupby(level=0).median()
        rb.rename('rb', inplace=True)
        return rb


class SupernovaeDetectionFeatureComputer(SingleBandFeatureExtractor):
    def _compute_features(self, detections):
        if len(detections.index.unique()) > 1:
            raise Exception('SupernovaeDetectionFeatureComputer handles one lightcurve at a time')

        columns = [
            'n_det_fid',
            'n_neg',
            'n_pos',
            'delta_mjd_fid',
            'delta_mag_fid',
            'positive_fraction',
            'min_mag',
            'mean_mag',
            'first_mag',
        ]

        detections = detections.sort_values('mjd')

        count = len(detections)
        if count == 0:
            features = pd.DataFrame(
                columns=columns
            )
            features.index.name = 'oid'
            return features

        columns = ['n_pos']
        n_pos = len(detections[detections.isdiffpos > 0])
        n_neg = len(detections[detections.isdiffpos < 0])

        min_mag = detections['magpsf_corr'].min()
        first_mag = detections.iloc[0]['magpsf_corr']

        delta_mjd_fid = detections.iloc[count-1]['mjd'] - detections.iloc[0]['mjd']
        delta_mag_fid = detections['magpsf_corr'].max() - min_mag

        positive_fraction = n_pos/(n_pos + n_neg)
        mean_mag = detections['magpsf_corr'].mean()

        features = pd.DataFrame(
            [
                count,
                n_neg,
                n_pos,
                delta_mjd_fid,
                delta_mag_fid,
                positive_fraction,
                min_mag,
                mean_mag,
                first_mag
            ],
            columns=columns,
            index=detections.index.value[0]
        )
        features.index.name = 'oid'
        return features


class SupernovaeNonDetectionFeatureComputer(SingleBandFeatureExtractor):
    def _compute_features(self, non_detections):
        if len(non_detections.index.unique()) > 1:
            raise Exception('SupernovaeNonDetectionFeatureComputer handles one lightcurve at a time')

        columns = [
            'n_non_det_before_fid',
            'max_diffmaglim_before_fid',
            'median_diffmaglim_before_fid',
            'last_diffmaglim_before_fid',
            'last_mjd_before_fid',
            'dmag_non_det_fid',
            'dmag_first_det_fid'
        ]
        # TODO end implementation


class FeaturesComputerV2:
    def __init__(self, idx):
        self.idx = idx

        class DetectionsV2Preprocessor(DetectionsPreprocessor):
            def preprocess_detections(self, detections):
                detections = self.discard_invalid_value_detections(detections)
                detections = self.drop_duplicates(detections)
                detections = self.discard_noisy_detections(detections)
                return detections

        self.preprocessor = DetectionsV2Preprocessor()
        self.turbo_fats_feature_extractor = FeatureExtractorFromSingleLCExtractor(
            FeatureExtractorPerBand(
                TurboFatsFeatureExtractor()
            )
        )
        self.galactic_feature_extractor = GalacticCoordinatesComputer()
        self.feature_extractor = FeatureExtractorFromFEList(
            [self.turbo_fats_feature_extractor, self.galactic_feature_extractor]
        )

    def execute(self, detections, non_detections=None, verbose=None):
        detections = self.preprocessor.preprocess_detections(detections)
        features = self.feature_extractor.compute_features(detections)
        return features
