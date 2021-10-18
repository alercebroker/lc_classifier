from typing import Tuple, List
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from numba import jit


from ..core.base import FeatureExtractorSingleBand
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
                bounds=[[A_bounds[0], t0_bounds[0], gamma_bounds[0], beta_bounds[0], trise_bounds[0], tfall_bounds[0]],
                        [A_bounds[1], t0_bounds[1], gamma_bounds[1], beta_bounds[1], trise_bounds[1], tfall_bounds[1]]],
                ftol=A_guess / 20.)
        except (ValueError, RuntimeError, OptimizeWarning):
            try:
                pout, pcov = curve_fit(
                    model_inference,
                    times,
                    fluxpsf,
                    p0=p0,
                    bounds=[[A_bounds[0], t0_bounds[0], gamma_bounds[0], beta_bounds[0], trise_bounds[0], tfall_bounds[0]],
                            [A_bounds[1], t0_bounds[1], gamma_bounds[1], beta_bounds[1], trise_bounds[1], tfall_bounds[1]]],
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
