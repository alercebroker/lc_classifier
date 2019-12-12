"""
Feature computation for each object
-----------------------------------
"""

# Author: Ignacio Reyes (based on Pablo Huijse's code)

import numpy as np
import pandas as pd

from turbofats import NewFeatureSpace
from astropy.coordinates import SkyCoord
import logging

from IPython.core.debugger import set_trace


logger = logging.getLogger(__name__)


def build_dataframe(index, columns, values):
    df = pd.DataFrame(
        index=index,
        columns=columns,
        data=values)
    df.index.name = 'oid'
    return df


class FeaturesComputer:
    def __init__(self, idx):
        self.idx = idx
        self.bands = [1, 2]  # fid: filter id
        self.cols = ['magpsf_corr', 'mjd', 'sigmapsf_corr']
        self.gal_feat_list = ['Amplitude', 'AndersonDarling', 'Autocor_length',
                              'Beyond1Std', 'CAR_sigma', 'CAR_mean', 'CAR_tau',
                              'Con', 'Eta_e',
                              'Gskew',
                              'MaxSlope', 'Mean', 'Meanvariance', 'MedianAbsDev',
                              'MedianBRP', 'PairSlopeTrend', 'PercentAmplitude', 'Q31',
                              'PeriodLS',
                              'Period_fit', 'Psi_CS', 'Psi_eta', 'Rcs',
                              'Skew', 'SmallKurtosis', 'Std',
                              'StetsonK']

        # This features are slow, refactor?
        self.harmonics_features = []
        for f in range(3):
            for index in range(4):
                self.harmonics_features.append(
                    "Freq" + str(f + 1) + "_harmonics_amplitude_" + str(index))
                self.harmonics_features.append(
                    "Freq" + str(f + 1) + "_harmonics_rel_phase_" + str(index))

        self.single_band_features_keys = self.gal_feat_list + self.harmonics_features
        self.fats_fs = NewFeatureSpace(self.single_band_features_keys)
        self.full_feature_list = []
        for band in self.bands:
            self.full_feature_list.append('n_samples_%d' % band)
            self.full_feature_list += [f+'_'+str(band)
                                       for f in self.single_band_features_keys]
        self.full_feature_list += ['gal_b', 'gal_l']

    def preprocess_df(self, df_alerts, filter_corrupted_alerts=True):
        df_alerts = df_alerts.copy()
        # Corrupted alerts
        df_alerts.replace([np.inf, -np.inf], np.nan, inplace=True)
        valid_alerts = df_alerts[[
            'mjd','ra','dec','magpsf_corr','sigmapsf_corr']].notna().all(axis=1)
        if filter_corrupted_alerts:
            # Just keep valid alerts
            df_alerts = df_alerts[valid_alerts.values]
        else:
            # Raise exception if an alert is corrupted
            is_corrupted = df_alerts.all(axis=None)
            if is_corrupted:
                raise ValueError("Alerts contains some illegal value: \n%s" %
                                 repr(df_alerts[np.logical_not(valid_alerts.values)]))

        df_alerts['oid'] = df_alerts.index
        df_alerts.drop_duplicates(['mjd', 'oid'], inplace=True)
        df_alerts = df_alerts[
            (df_alerts.sigmapsf_corr > 0.0)
            & (df_alerts.sigmapsf_corr < 1.0)
            & (df_alerts.rb >= 0.55)]
        df_alerts = df_alerts[[col for col in df_alerts.columns if col != 'oid']]
        return df_alerts

    def execute(self, df_alerts, filter_corrupted_alerts=True):
        logger.info("Features %d: Calculating Features" % (self.idx))
        df_alerts = df_alerts.copy()
        index = []
        columns = []
        for band in self.bands:
            columns.append('n_samples_%d' % band)
            columns += [feature_name+'_%d' %
                        band for feature_name in self.fats_fs.feature_names]
        columns += ['gal_b', 'gal_l']
        values = []

        df_alerts = self.preprocess_df(
            df_alerts, filter_corrupted_alerts=filter_corrupted_alerts)

        oids = df_alerts.index.unique()
        for oid in oids:
            try:
                object_alerts = df_alerts.loc[[oid]]
                enough_alerts = (
                    len(object_alerts) > 0
                    and (len(object_alerts[object_alerts.fid == 1]) > 5)
                    and (len(object_alerts[object_alerts.fid == 2]) > 5))

                if not enough_alerts:
                    continue

                object_features = []
                for band in self.bands:
                    data = object_alerts.query("fid == %d" % band)
                    n_samples = len(data)
                    object_features.append(n_samples)
                    features = self.fats_fs.calculate_features(data)
                    object_features += features.values.flatten().tolist()
                values.append(object_features)
                index.append(oid)

            except:
                logging.exception('Exception computing features with oid={}'.format(oid) )
                continue

        if len(index) == 0:
            df = pd.DataFrame(columns=columns)
            df.index.name = 'oid'
            return df

        values = np.array(values)
        radec_df = df_alerts.loc[index][['ra', 'dec']].groupby('oid').mean()
        radec_df = radec_df.loc[index]
        radec = SkyCoord(
            ra=radec_df['ra'], dec=radec_df['dec'], frame='icrs', unit='deg')
        galactic = radec.galactic
        np_galactic = np.stack((galactic.b.degree, galactic.l.degree), axis=-1)
        values = np.concatenate((values, np_galactic), axis=1)
        df = build_dataframe(index, columns, values)
        # some features might be NaN
        are_nans = df.isna()
        if are_nans.any(axis=None):
            wrong_oids = df[are_nans.any(axis='columns')].index
            logging.warning('Dropping objects with NaN features: %s' % repr(wrong_oids))
        return df.dropna()


class FeaturesComputerV2(FeaturesComputer):
    def __init__(self, idx):
        self.idx = idx
        self.bands = [1, 2]  # fid: filter id
        self.single_band_features_keys = [
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
            'IAR_phi', #'FluxPercentileRatioMid20', 'FluxPercentileRatioMid35', 'FluxPercentileRatioMid50', 'FluxPercentileRatioMid65', 'FluxPercentileRatioMid80',
            'LinearTrend'
        ]

        self.fats_fs = NewFeatureSpace(self.single_band_features_keys)
        self.full_feature_list = []
        for band in self.bands:
            self.full_feature_list.append('n_samples_%d' % band)
            self.full_feature_list += [f+'_'+str(band) for f in self.single_band_features_keys]
        self.full_feature_list += ['gal_b', 'gal_l']

    def execute(self, df_alerts, filter_corrupted_alerts=True, verbose=1):
        logger.info("Features %d: Calculating Features" % (self.idx))
        df_alerts = df_alerts.copy()
        n_original_alerts = len(df_alerts)
        n_original_objects = len(df_alerts.index.unique())
        index = []
        columns = []
        for band in self.bands:
            columns.append('n_samples_%d' % band)
            columns += [feature_name+'_%d' %
                        band for feature_name in self.fats_fs.feature_names]
        columns += ['gal_b', 'gal_l']
        values = []

        df_alerts = self.preprocess_df(
            df_alerts, filter_corrupted_alerts=filter_corrupted_alerts)

        oids = df_alerts.index.unique()
        n_objects_after_preprocessing = len(oids)
        n_alerts_after_preprocessing = len(df_alerts)
        for oid in oids:
            try:
                object_alerts = df_alerts.loc[[oid]]
                enough_alerts = (
                    len(object_alerts) > 0
                    and (
                        (len(object_alerts[object_alerts.fid == 1]) > 5)
                    or (len(object_alerts[object_alerts.fid == 2]) > 5)
                    )
                )

                if not enough_alerts:
                    continue

                object_features = []
                for band in self.bands:
                    n_samples = len(object_alerts[object_alerts.fid == band])
                    object_features.append(n_samples)
                    enough_alerts_band = n_samples > 5
                    if enough_alerts_band:
                        data = object_alerts.query("fid == %d" % band)
                        features = self.fats_fs.calculate_features(data)
                        object_features += features.values.flatten().tolist()
                    else:
                        object_features += [np.nan]*len(self.fats_fs.feature_names)
                values.append(object_features)
                index.append(oid)

            except:
                if verbose > 0:
                    logging.exception('Exception computing features with oid={}'.format(oid) )
                continue

        if len(index) == 0:
            df = pd.DataFrame(columns=columns)
            df.index.name = 'oid'
            if verbose >= 3:
                logging.info(
                    f"[Features {self.idx}] "
                    f"Input {n_original_objects} objects, {n_original_alerts} alerts. "
                    f"After preprocessing {n_objects_after_preprocessing} objects, "
                    f"{n_alerts_after_preprocessing}. "
                    f"Output of zero objects"
                )
            return df

        values = np.array(values)
        radec_df = df_alerts.loc[index][['ra', 'dec']].groupby('oid').mean()
        radec_df = radec_df.loc[index]
        radec = SkyCoord(
            ra=radec_df['ra'], dec=radec_df['dec'], frame='icrs', unit='deg')
        galactic = radec.galactic
        np_galactic = np.stack((galactic.b.degree, galactic.l.degree), axis=-1)
        values = np.concatenate((values, np_galactic), axis=1)
        df = build_dataframe(index, columns, values)
        # some features might be NaN
        if verbose >= 3:
            are_nans = df.isna()
            if are_nans.any(axis=None):
                wrong_oids = df[are_nans.any(axis='columns')].index
            logging.info(
                f"[Features {self.idx}] "
                f"Input {n_original_objects} objects, {n_original_alerts} alerts. "
                f"After preprocessing {n_objects_after_preprocessing} objects, "
                f"{n_alerts_after_preprocessing}. "
                f"Output {len(df)} objects, {len(wrong_oids)} of them with NaNs"
            )
        return df


def compute_det_features(det,fid):

    det_fid     = det[ det.fid == fid ]

    count = len(det_fid)
    result = { 'n_det_fid'                     : count,
               'n_neg'                         : np.nan,
               'delta_mjd_fid'                 : np.nan,
               'delta_mag_fid'                 : np.nan,
               'positive_fraction'             : np.nan,
               'min_mag'                                : np.nan,
               'mean_mag'                               : np.nan,
               'first_mag'                              : np.nan,
               'rb'                                     : np.nan
    }

    if count == 0:
        return result

    n_pos = len(det_fid[ det_fid.isdiffpos > 0 ])
    n_neg = len(det_fid[ det_fid.isdiffpos < 0 ])

    min_mag   = det_fid['magpsf_corr'].min()
    first_mag = det_fid.iloc[0]['magpsf_corr']

    result2 = {
               'n_pos'                         : n_pos,
               'n_neg'                         : n_neg,
               'delta_mjd_fid'                 : det_fid.iloc[count-1]['mjd'] - det_fid.iloc[0]['mjd'],
               'delta_mag_fid'                 : det_fid['magpsf_corr'].max() - min_mag,
               'positive_fraction'             : n_pos/(n_pos+n_neg),
               'min_mag'                                : min_mag,
               'mean_mag'                               : det_fid['magpsf_corr'].mean(),
               'first_mag'                              : first_mag,
    }
    result.update(result2)

    return result


def compute_before_features(det_result,non_det,fid):

    non_det_fid  = non_det[ non_det.fid == fid ]

    count = len(non_det_fid)
    result = {
                'n_non_det_before_fid'             : count,
                'max_diffmaglim_before_fid'        : np.nan,
                'median_diffmaglim_before_fid'     : np.nan,
                'last_diffmaglim_before_fid'       : np.nan,
                'last_mjd_before_fid'              : np.nan,
                'dmag_non_det_fid'                 : np.nan,
                'dmag_first_det_fid'               : np.nan
    }

    if len(non_det_fid) == 0:
        return result

    last_element = non_det_fid.iloc[count-1]

    median_diffmaglim_before = non_det_fid['diffmaglim'].median()

    result2 = {
                'max_diffmaglim_before_fid'        : non_det_fid['diffmaglim'].max(),
                'median_diffmaglim_before_fid'     : median_diffmaglim_before,
                'last_diffmaglim_before_fid'       : last_element['diffmaglim'],
                'last_mjd_before_fid'              : last_element['mjd'],
                'dmag_non_det_fid'                 : median_diffmaglim_before - det_result['min_mag'],
                'dmag_first_det_fid'               : last_element['diffmaglim'] - det_result['first_mag']
    }

    result.update(result2)

    return result

def compute_after_features(non_det,fid):

    non_det_fid = non_det[ non_det.fid == fid ]

    count = len(non_det_fid)
    result = {
               'n_non_det_after_fid'         : count,
               'max_diffmaglim_after_fid'    : np.nan,
               'median_diffmaglim_after_fid' : np.nan
    }
    if count == 0:
        return result

    result2 = {
               'max_diffmaglim_after_fid'    : non_det_fid['diffmaglim'].max(),
               'median_diffmaglim_after_fid' : non_det_fid['diffmaglim'].median()
    }

    result.update(result2)

    return result



def calculate_hacked_features(detections,non_detections,band,first_mjd,features):

    detections = detections.sort_values("mjd")
    non_detections = non_detections.sort_values("mjd")


    results_det = compute_det_features(detections,band)


    non_detections_before = non_detections[non_detections.mjd < first_mjd]
    results_before = compute_before_features(results_det,non_detections_before,band)


    non_detections_after = non_detections[non_detections.mjd >= first_mjd]
    results_after = compute_after_features(non_detections_after,band)


    all_results = {
        **results_det,
        **results_before,
        **results_after
    }
    final_features = []


    for feature in features:
        final_features.append(float(all_results[feature]))

    return final_features


class FeaturesComputerHierarchical(FeaturesComputer):
    def __init__(self, idx):
        self.idx = idx
        self.bands = [1, 2]  # fid: filter id
        self.single_band_features_keys = [
            'Amplitude', 'AndersonDarling', 'Autocor_length',
            'Beyond1Std',
            'Con', 'Eta_e',
            'Gskew',
            'MaxSlope',  'Meanvariance', 'MedianAbsDev',
            'MedianBRP', 'PairSlopeTrend', 'PercentAmplitude', 'Q31',
            'PeriodLS_v2',
            'Period_fit_v2', 'Psi_CS_v2', 'Psi_eta_v2', 'Rcs',
            'Skew', 'SmallKurtosis', 'Std',
            'StetsonK', 'Harmonics',
            'Pvar', 'ExcessVar',
            'GP_DRW_sigma', 'GP_DRW_tau', 'SF_ML_amplitude', 'SF_ML_gamma',
            'IAR_phi',
            'LinearTrend'
        ]

        self.fats_fs = NewFeatureSpace(self.single_band_features_keys)
        self.full_feature_list = []
        for band in self.bands:
            self.full_feature_list.append('n_samples_%d' % band)
            self.full_feature_list += [f+'_'+str(band) for f in self.single_band_features_keys]
        self.full_feature_list += ['gal_b', 'gal_l','sgscore1','g-r_max','g-r_mean']

    def execute(self, df_alerts, df_non_detections, filter_corrupted_alerts=True, verbose=1):
        logger.info("Features %d: Calculating Features" % (self.idx))

        df_alerts = df_alerts.copy()
        df_non_detections = df_non_detections.copy()

        n_original_alerts = len(df_alerts)
        n_original_objects = len(df_alerts.index.unique())
        index = []
        columns = []

        non_detections_features = ['delta_mag_fid',
                                    'delta_mjd_fid',
                                    'dmag_first_det_fid',
                                    'dmag_non_det_fid',
                                    'last_diffmaglim_before_fid',
                                    'last_mjd_before_fid',
                                    'max_diffmaglim_after_fid',
                                    'max_diffmaglim_before_fid',
                                    'median_diffmaglim_after_fid',
                                    'median_diffmaglim_before_fid',
                                    'n_non_det_after_fid',
                                    'n_non_det_before_fid',
                                    'positive_fraction' ]

        for band in self.bands:
            columns.append('n_samples_%d' % band)
            columns += [feature_name+'_%d' %
                        band for feature_name in self.fats_fs.feature_names + non_detections_features]

        columns += ['gal_b', 'gal_l','sgscore1','g-r_max','g-r_mean', 'rb']



        values = []

        df_alerts = self.preprocess_df(
            df_alerts, filter_corrupted_alerts=filter_corrupted_alerts)

        oids = df_alerts.index.unique()
        n_objects_after_preprocessing = len(oids)
        n_alerts_after_preprocessing = len(df_alerts)
        for oid in oids:
            try:
                object_alerts = df_alerts.loc[oid]
                enough_alerts = (
                    len(object_alerts) > 0
                    and type(object_alerts) is pd.DataFrame
                    and (
                        (len(object_alerts[object_alerts.fid == 1]) > 5)
                    or (len(object_alerts[object_alerts.fid == 2]) > 5)
                    )
                )

                if not enough_alerts:
                    continue

                first_mjd = object_alerts.sort_values("mjd").iloc[0].mjd

                try:
                    object_non_detections = df_non_detections.loc[oid]
                except KeyError:
                    object_non_detections = pd.DataFrame(columns=["fid","mjd"])


                object_features = []
                for band in self.bands:
                    n_samples = len(object_alerts[object_alerts.fid == band])
                    object_features.append(n_samples)
                    enough_alerts_band =  n_samples > 5
                    data = object_alerts.query("fid == %d" % band)
                    data_non_detections = object_non_detections.query("fid == %d" % band)
                    if enough_alerts_band:
                            features = self.fats_fs.calculate_features(data)
                            object_features += features.values.flatten().tolist()
                    else:
                            object_features += [np.nan]*len(self.fats_fs.feature_names)
                    object_hacked_features = calculate_hacked_features(data,data_non_detections,band,first_mjd,non_detections_features)
                    object_features = object_features+object_hacked_features
                values.append(object_features)
                index.append(oid)

            except:
                raise
                if verbose > 0:
                    logging.exception('Exception computing features with oid={}'.format(oid) )
                continue

        if len(index) == 0:
            df = pd.DataFrame(columns=columns)
            df.index.name = 'oid'
            if verbose >= 3:
                logging.info(
                    f"[Features {self.idx}] "
                    f"Input {n_original_objects} objects, {n_original_alerts} alerts. "
                    f"After preprocessing {n_objects_after_preprocessing} objects, "
                    f"{n_alerts_after_preprocessing}. "
                    f"Output of zero objects"
                )
            return df

        values = np.array(values)
        radec_df = df_alerts.loc[index][['ra', 'dec']].groupby('oid').mean()
        radec_df = radec_df.loc[index]
        radec = SkyCoord(
            ra=radec_df['ra'], dec=radec_df['dec'], frame='icrs', unit='deg')
        galactic = radec.galactic

        np_galactic = np.stack((galactic.b.degree, galactic.l.degree), axis=-1)
        sgscore = df_alerts.sgscore1.groupby("oid").median()#.values
        g_r_max = (df_alerts[df_alerts.fid == 1 ]["magpsf_corr"].groupby("oid").min() - df_alerts[df_alerts.fid == 2 ]["magpsf_corr"].groupby("oid").min())#.values
        g_r_max.rename("g-r_max",inplace=True)
        g_r_mean = (df_alerts[df_alerts.fid == 1 ]["magpsf_corr"].groupby("oid").mean() - df_alerts[df_alerts.fid == 2 ]["magpsf_corr"].groupby("oid").mean())#.values
        g_r_mean.rename("g-r_mean",inplace=True)

        rb = df_alerts.rb.groupby("oid").median()
        rb.rename("rb",inplace=True)

        values = np.concatenate((values, np_galactic),axis=1)

        df = pd.DataFrame(values, index=index)
        df = df.join(sgscore)
        df = df.join(g_r_max)
        df = df.join(g_r_mean)
        df = df.join(rb)
        df.columns = columns
        df["oid"] = df.index

        # df = build_dataframe(index, columns, values)
        # some features might be NaN
        if verbose >= 3:
            are_nans = df.isna()
            if are_nans.any(axis=None):
                wrong_oids = df[are_nans.any(axis='columns')].index
            logging.info(
                f"[Features {self.idx}] "
                f"Input {n_original_objects} objects, {n_original_alerts} alerts. "
                f"After preprocessing {n_objects_after_preprocessing} objects, "
                f"{n_alerts_after_preprocessing}. "
                f"Output {len(df)} objects, {len(wrong_oids)} of them with NaNs"
            )
        return df
