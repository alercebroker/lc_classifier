from typing import Tuple, List
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize
from numba import jit


from ..core.base import FeatureExtractorSingleBand, FeatureExtractor
from scipy.optimize import OptimizeWarning
import logging

from ...utils import mag_to_flux


@jit(nopython=True)
def model_inference(times, A, t0, gamma, beta, t_rise, t_fall):
    sigmoid_factor = 1.0 / 3.0
    t1 = t0 + gamma

    sigmoid = 1.0 / (1.0 + np.exp(-sigmoid_factor * (times - t1)))
    den = 1 + np.exp(-(times - t0) / t_rise)
    flux = ((1 - beta) * np.exp(-(times - t1) / t_fall)
            * sigmoid
            + (1. - beta * (times - t0) / gamma)
            * (1 - sigmoid)) * A / den
    return flux


class SNModelScipy(object):
    def __init__(self):
        self.parameters = None

    def fit(self, times, targets, obs_errors):
        """Assumptions:
            min(times) == 0"""
        fluxpsf = targets

        # Parameter bounds
        argmax_fluxpsf = np.argmax(fluxpsf)
        max_fluxpsf = fluxpsf[argmax_fluxpsf]
        A_bounds = [max_fluxpsf / 3.0, max_fluxpsf * 3.0]
        t0_bounds = [-50.0, 50.0]
        gamma_bounds = [1.0, 100.0]
        beta_bounds = [0.0, 1.0]
        trise_bounds = [1.0, 100.0]
        tfall_bounds = [1.0, 100.0]

        # Parameter guess
        A_guess = max(A_bounds[1], max(A_bounds[0], 1.2 * max_fluxpsf))
        t0_guess = -5.0
        gamma_guess = min(gamma_bounds[1], max(
            gamma_bounds[0], max(times)))
        beta_guess = 0.5
        trise_guess = min(
            trise_bounds[1],
            max(trise_bounds[0], times[argmax_fluxpsf] / 2.0))
        tfall_guess = 40.0

        # reference guess
        p0 = [A_guess, t0_guess, gamma_guess,
              beta_guess, trise_guess, tfall_guess]

        # get parameters
        try:
            pout, pcov = curve_fit(
                model_inference,
                times,
                fluxpsf,
                p0=p0,
                bounds=[
                    [
                        A_bounds[0],
                        t0_bounds[0],
                        gamma_bounds[0],
                        beta_bounds[0],
                        trise_bounds[0],
                        tfall_bounds[0]
                    ],
                    [
                        A_bounds[1],
                        t0_bounds[1],
                        gamma_bounds[1],
                        beta_bounds[1],
                        trise_bounds[1],
                        tfall_bounds[1]
                    ]
                ],
                ftol=A_guess / 20.)
        except (ValueError, RuntimeError, OptimizeWarning):
            try:
                pout, pcov = curve_fit(
                    model_inference,
                    times,
                    fluxpsf,
                    p0=p0,
                    bounds=[
                        [
                            A_bounds[0],
                            t0_bounds[0],
                            gamma_bounds[0],
                            beta_bounds[0],
                            trise_bounds[0],
                            tfall_bounds[0]
                        ],
                        [
                            A_bounds[1],
                            t0_bounds[1],
                            gamma_bounds[1],
                            beta_bounds[1],
                            trise_bounds[1],
                            tfall_bounds[1]
                        ]
                    ],
                    ftol=A_guess / 3.)
            except (ValueError, RuntimeError, OptimizeWarning):
                pout = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

        self.parameters = pout
        predictions = model_inference(
            times,
            self.parameters[0],
            self.parameters[1],
            self.parameters[2],
            self.parameters[3],
            self.parameters[4],
            self.parameters[5])

        # mse = np.mean((predictions - targets) ** 2)
        # nmse = mse / np.mean(targets ** 2)

        chi = np.sum((predictions - targets) ** 2 / (obs_errors + 0.01) ** 2)
        chi_den = len(predictions) - len(self.parameters)
        if chi_den >= 1:
            chi_per_degree = chi / chi_den
        else:
            chi_per_degree = np.NaN
        return chi_per_degree

    def get_model_parameters(self):
        return self.parameters.tolist()


class SNModelScipyPhaseII(object):
    def __init__(self):
        self.parameters = None

    def fit(self, times, fluxpsf, obs_errors):
        """Assumptions:
            min(times) == 0"""

        # Parameter bounds
        argmax_fluxpsf = np.argmax(fluxpsf)
        max_fluxpsf = fluxpsf[argmax_fluxpsf]
        A_bounds = [max_fluxpsf / 3.0, max_fluxpsf * 3.0]
        t0_bounds = [-50.0, np.max(times)]
        gamma_bounds = [1.0, 120.0]
        beta_bounds = [0.0, 1.0]
        trise_bounds = [1.0, 100.0]
        tfall_bounds = [1.0, 180.0]

        # Parameter guess
        A_guess = np.clip(1.2 * max_fluxpsf, A_bounds[0], A_bounds[1])
        t0_guess = np.clip(times[argmax_fluxpsf] - 10, t0_bounds[0], t0_bounds[1])
        gamma_guess = 3.0
        beta_guess = 0.5
        trise_guess = 3.0
        tfall_guess = 50.0

        # reference guess
        p0 = [A_guess, t0_guess, gamma_guess,
              beta_guess, trise_guess, tfall_guess]

        # get parameters
        try:
            pout, pcov = curve_fit(
                f=model_inference,
                xdata=times,
                ydata=fluxpsf,
                p0=p0,
                sigma=5+obs_errors,
                bounds=[[A_bounds[0], t0_bounds[0], gamma_bounds[0], beta_bounds[0], trise_bounds[0], tfall_bounds[0]],
                        [A_bounds[1], t0_bounds[1], gamma_bounds[1], beta_bounds[1], trise_bounds[1], tfall_bounds[1]]],
                ftol=1e-8,
                # verbose=2
            )
        except (ValueError, RuntimeError, OptimizeWarning):
            try:
                logging.info('First fit of SPM failed. Attempting second fit.')
                pout, pcov = curve_fit(
                    f=model_inference,
                    xdata=times,
                    ydata=fluxpsf,
                    p0=p0,
                    sigma=5+obs_errors,
                    bounds=[[A_bounds[0], t0_bounds[0], gamma_bounds[0], beta_bounds[0], trise_bounds[0], tfall_bounds[0]],
                            [A_bounds[1], t0_bounds[1], gamma_bounds[1], beta_bounds[1], trise_bounds[1], tfall_bounds[1]]],
                    ftol=0.1,
                    # verbose=2
                )
            except (ValueError, RuntimeError, OptimizeWarning):
                logging.info('Two fits of SPM failed. Returning NaN.')
                pout = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

        self.parameters = pout
        predictions = model_inference(
            times,
            self.parameters[0],
            self.parameters[1],
            self.parameters[2],
            self.parameters[3],
            self.parameters[4],
            self.parameters[5])

        # mse = np.mean((predictions - targets) ** 2)
        # nmse = mse / np.mean(targets ** 2)

        chi = np.sum((predictions - fluxpsf) ** 2 / (obs_errors + 0.01) ** 2)
        chi_den = len(predictions) - len(self.parameters)
        if chi_den >= 1:
            chi_per_degree = chi / chi_den
        else:
            chi_per_degree = np.NaN
        return chi_per_degree

    def get_model_parameters(self):
        return self.parameters.tolist()


class SNParametricModelExtractor(FeatureExtractorSingleBand):
    """Fits a SNe parametric model to the light curve and provides
    the fitted parameters as features."""

    def __init__(self, bands: List):
        super().__init__(bands)
        self.sn_model = SNModelScipy()

    @lru_cache(1)
    def get_features_keys_without_band(self) -> Tuple[str, ...]:
        return (
            'SPM_A',
            'SPM_t0',
            'SPM_gamma',
            'SPM_beta',
            'SPM_tau_rise',
            'SPM_tau_fall',
            'SPM_chi'
        )

    @lru_cache(1)
    def get_required_keys(self) -> Tuple[str, ...]:
        return 'time', 'magpsf', 'sigmapsf', 'band'

    def compute_feature_in_one_band(self, detections, band, **kwargs):
        grouped_detections = detections.groupby(level=0)
        return self.compute_feature_in_one_band_from_group(grouped_detections, band=band, **kwargs)

    def compute_feature_in_one_band_from_group(self, detections, band, **kwargs):
        columns = self.get_features_keys_with_band(band)

        def aux_function(oid_detections, **kwargs):
            if band not in oid_detections['band'].values:
                oid = oid_detections.index.values[0]
                logging.debug(
                    f'extractor=SN parametric model object={oid} required_cols={self.get_required_keys()} band={band}')
                return self.nan_series_in_band(band)

            oid_band_detections = oid_detections[oid_detections['band'] == band]

            oid_band_detections = oid_band_detections[[
                'time', 'magpsf', 'sigmapsf']]
            oid_band_detections = oid_band_detections.dropna()

            times = oid_band_detections['time'].values
            times = times - np.min(times)
            mag_targets = oid_band_detections['magpsf'].values
            targets = mag_to_flux(mag_targets)
            errors = oid_band_detections['sigmapsf'].values
            errors = mag_to_flux(mag_targets - errors) - targets

            times = times.astype(np.float32)
            targets = targets.astype(np.float32)

            fit_error = self.sn_model.fit(times, targets, errors)
            model_parameters = self.sn_model.get_model_parameters()
            model_parameters.append(fit_error)

            out = pd.Series(
                data=model_parameters,
                index=columns
            )
            return out

        sn_params = detections.apply(aux_function)
        sn_params.index.name = 'oid'
        return sn_params


class SPMExtractorPhaseII(FeatureExtractorSingleBand):
    """Fits a SNe parametric model to the light curve and provides
    the fitted parameters as features."""

    def __init__(self, bands: List):
        super().__init__(bands)
        self.sn_model = SNModelScipyPhaseII()

    @lru_cache(1)
    def get_features_keys_without_band(self) -> Tuple[str, ...]:
        return (
            'SPM_A',
            'SPM_t0',
            'SPM_gamma',
            'SPM_beta',
            'SPM_tau_rise',
            'SPM_tau_fall',
            'SPM_chi'
        )

    @lru_cache(1)
    def get_required_keys(self) -> Tuple[str, ...]:
        return 'time', 'difference_flux', 'difference_flux_error', 'band'

    def compute_feature_in_one_band(self, detections, band, **kwargs):
        grouped_detections = detections.groupby(level=0)
        return self.compute_feature_in_one_band_from_group(
            grouped_detections, band=band, **kwargs)

    def compute_feature_in_one_band_from_group(self, detections, band, **kwargs):
        columns = self.get_features_keys_with_band(band)

        def aux_function(oid_detections, **kwargs):
            if band not in oid_detections['band'].values:
                oid = oid_detections.index.values[0]
                logging.debug(
                    f'extractor=SN parametric model object={oid} required_cols={self.get_required_keys()} band={band}')
                return self.nan_series_in_band(band)

            oid_band_detections = oid_detections[oid_detections['band'] == band]

            oid_band_detections = oid_band_detections[[
                'time', 'difference_flux', 'difference_flux_error']]
            oid_band_detections = oid_band_detections.dropna()

            np_array_data = oid_band_detections.values.astype(np.float32)
            times = np_array_data[:, 0]
            times = times - np.min(times)
            flux_target = np_array_data[:, 1]
            errors = np_array_data[:, 2]

            fit_error = self.sn_model.fit(times, flux_target, errors)
            model_parameters = self.sn_model.get_model_parameters()
            model_parameters.append(fit_error)

            out = pd.Series(
                data=model_parameters,
                index=columns
            )
            return out

        sn_params = detections.apply(aux_function)
        sn_params.index.name = 'oid'
        return sn_params


class SPMExtractorElasticc(FeatureExtractor):
    """Fits a SNe parametric model to the light curve and provides
    the fitted parameters as features."""

    def __init__(self, bands: List):
        self.bands = bands
        self.sn_model = SNModelScipyElasticc(self.bands)

    @lru_cache(1)
    def get_features_keys(self) -> Tuple[str, ...]:
        features = []
        model_params = [
            'SPM_A',
            'SPM_t0',
            'SPM_gamma',
            'SPM_beta',
            'SPM_tau_rise',
            'SPM_tau_fall'
        ]
        for band in self.bands:
            for feature in model_params:
                features.append(feature+'_'+band)

        features += [f'SPM_chi_{band}' for band in self.bands]
        return features

    @lru_cache(1)
    def get_required_keys(self) -> Tuple[str, ...]:
        return 'time', 'difference_flux', 'difference_flux_error', 'band'

    def _compute_features(self, detections, **kwargs):
        return self._compute_features_from_df_groupby(
            detections.groupby(level=0),
            **kwargs)

    def _compute_features_from_df_groupby(
            self, detections, **kwargs) -> pd.DataFrame:
        columns = self.get_features_keys()
        
        def aux_function(oid_detections, **kwargs):
            oid = oid_detections.index.values[0]
            bands = oid_detections['band'].values
            available_bands = np.unique(bands)

            np_array_data = oid_detections[
                ['time',
                 'difference_flux',
                 'difference_flux_error',
                 'band']].values

            times = np_array_data[:, 0]
            times = times - np.min(times)
            flux_target = np_array_data[:, 1]
            errors = np_array_data[:, 2]
            bands = np_array_data[:, 3]
            
            self.sn_model.fit(times, flux_target, errors, bands)

            model_parameters = self.sn_model.get_model_parameters()
            chis = self.sn_model.get_chis()

            out = pd.Series(
                data=np.concatenate([model_parameters, chis], axis=0),
                index=columns
            )
            return out

        sn_params = detections.apply(aux_function)
        sn_params.index.name = 'oid'
        return sn_params


@jit(nopython=True)
def objective_function(params, times, fluxpsf, obs_errors, bands, available_bands):
    params = params.reshape(-1, 6)
    sum_sqerrs = 0.0
    for i, band in enumerate(available_bands):
        band_mask = bands == band
        band_times = times[band_mask]
        band_flux = fluxpsf[band_mask]
        band_errors = obs_errors[band_mask]

        band_params = params[i]

        A = band_params[0]
        t0 = band_params[1]
        gamma = band_params[2]
        beta = band_params[3]
        t_rise = band_params[4]
        t_fall = band_params[5]

        sigmoid_factor = 1.0 / 3.0
        t1 = t0 + gamma

        sigmoid = 1.0 / (1.0 + np.exp(-sigmoid_factor * (band_times - t1)))
        den = 1 + np.exp(-(band_times - t0) / t_rise)
        model_output = ((1 - beta) * np.exp(-(band_times - t1) / t_fall)
                        * sigmoid
                        + (1. - beta * (band_times - t0) / gamma)
                        * (1 - sigmoid)) * A / den

        band_sqerr = np.sum(((model_output - band_flux) / (band_errors + 5.0)) ** 2)
        sum_sqerrs += band_sqerr

    params_var = np.empty(6, dtype=np.float32)
    params_var[0] = np.std(params[:, 0])
    params_var[1] = np.var(params[:, 1])
    params_var[2] = np.var(params[:, 2])
    params_var[3] = np.var(params[:, 3])
    params_var[4] = np.var(params[:, 4])
    params_var[5] = np.var(params[:, 5])

    lambdas = np.array([
        1.0 / np.mean(params[:, 0]),
        20.0 / 10 ** 2,
        3.0 / 10 ** 2,
        1.0 / 1 ** 2,
        1.0 / 10 ** 2,
        1.0 / 30 ** 2
    ], dtype=np.float32)

    loss = sum_sqerrs + np.dot(lambdas, params_var)
    return loss


class SNModelScipyElasticc(object):
    def __init__(self, bands):
        self.parameters = None
        self.chis = None
        self.bands = bands

    def fit(self, times, fluxpsf, obs_errors, bands):
        """Assumptions:
            min(times) == 0"""

        # Parameter bounds
        argmax_fluxpsf = np.argmax(fluxpsf)
        max_fluxpsf = fluxpsf[argmax_fluxpsf]
        
        A_bounds = [np.abs(max_fluxpsf)/10.0, np.abs(max_fluxpsf)*10.0]
        t0_bounds = [-50.0, np.max(times)]
        gamma_bounds = [1.0, 120.0]
        beta_bounds = [0.0, 1.0]
        trise_bounds = [1.0, 100.0]
        tfall_bounds = [1.0, 180.0]

        # Available bands
        available_bands = np.unique(bands)

        initial_guess = []
        
        for band in available_bands:
            band_mask = bands == band
            band_flux = fluxpsf[band_mask]
            
            # Parameter guess
            A_guess = np.clip(1.2*np.max(band_flux), A_bounds[0], A_bounds[1])
            t0_guess = np.clip(times[np.argmax(fluxpsf)], t0_bounds[0], t0_bounds[1])
            gamma_guess = 3.0
            beta_guess = 0.5
            trise_guess = 3.0
            tfall_guess = 50.0
        
            # reference guess
            p0 = [A_guess, t0_guess, gamma_guess,
                  beta_guess, trise_guess, tfall_guess]

            initial_guess.append(p0)
        initial_guess = np.concatenate(initial_guess, axis=0).astype(np.float32)

        bounds = [
            A_bounds,
            t0_bounds,
            gamma_bounds,
            beta_bounds,
            trise_bounds,
            tfall_bounds
        ]*len(available_bands)

        band_mapper = dict(zip('ugrizY', range(6)))

        res = minimize(
            objective_function,
            initial_guess,
            args=(
                times.astype(np.float32),
                fluxpsf.astype(np.float32),
                obs_errors.astype(np.float32),
                np.vectorize(band_mapper.get)(bands),
                np.vectorize(band_mapper.get)(available_bands)),
            method='L-BFGS-B',
            bounds=bounds,
            options={'iprint': -1, 'maxcor': 10, 'maxiter': 200}
        )
        success = res.success
        best_params = res.x.reshape(-1, 6)

        parameters = []
        chis = []
        for band in self.bands:
            if band in available_bands:
                index = available_bands.tolist().index(band)
                parameters.append(best_params[index, :])

                band_mask = bands == band
                band_times = times[band_mask]
                band_flux = fluxpsf[band_mask]
                band_errors = obs_errors[band_mask]

                predictions = model_inference(
                    band_times.astype(np.float32),
                    *best_params[index, :])

                chi = np.sum((predictions - band_flux) ** 2 / (band_errors + 5) ** 2)
                chi_den = len(predictions) - 6
                if chi_den >= 1:
                    chi_per_degree = chi / chi_den
                else:
                    chi_per_degree = np.NaN
                chis.append(chi_per_degree)
            else:
                parameters.append([np.nan]*6)
                chis.append(np.nan)
                
        self.parameters = np.concatenate(parameters, axis=0)
        self.chis = np.array(chis)

    def get_model_parameters(self):
        return self.parameters.tolist()

    def get_chis(self):
        return self.chis.tolist()
