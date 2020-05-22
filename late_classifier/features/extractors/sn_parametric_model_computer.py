from typing import List

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from ..core.base import FeatureExtractorSingleBand
from scipy.optimize import OptimizeWarning
import logging


class SNModelScipy(object):
    def __init__(self):
        self.beta = 1.0 / 3.0
        self.parameters = None

    def model(self, times, A, t0, gamma, f, t_rise, t_fall):
        t1 = t0 + gamma

        sigmoid = 1.0 / (1.0 + np.exp(-self.beta * (times - t1)))
        den = 1 + np.exp(-(times - t0) / t_rise)
        flux = (A * (1 - f) * np.exp(-(times - t1) / t_fall) / den
                * sigmoid
                + A * (1. - f * (times - t0) / gamma) / den
                * (1 - sigmoid))
        return flux

    def fit(self, times, targets, obs_errors):
        fluxpsf = targets
        mjds = times

        # Parameter bounds
        A_bounds = [max(fluxpsf) / 3.0, max(fluxpsf) * 3.0]
        t0_bounds = [-50.0, 50.0]
        gamma_bounds = [1.0, 100.0]
        f_bounds = [0.0, 1.0]
        trise_bounds = [1.0, 100.0]
        tfall_bounds = [1.0, 100.0]

        # Parameter guess
        A_guess = max(A_bounds[1], max(A_bounds[0], 1.2 * max(fluxpsf)))
        t0_guess = -5.0  # min(t0_bounds[1], max(t0_bounds[0], times[np.argmax(fluxpsf)]))
        gamma_guess = min(gamma_bounds[1], max(gamma_bounds[0], (times.max() - times.min())))
        f_guess = 0.5
        trise_guess = min(
            trise_bounds[1],
            max(trise_bounds[0], (times[np.argmax(fluxpsf)] - times.min()) / 2.0))
        tfall_guess = 40.0

        # reference guess
        p0 = [A_guess, t0_guess, gamma_guess, f_guess, trise_guess, tfall_guess]

        # update times and flux
        times = mjds - min(mjds)

        # get parameters
        try:
            pout, pcov = curve_fit(
                self.model,
                times,
                fluxpsf,
                p0=[A_guess, t0_guess, gamma_guess, f_guess, trise_guess, tfall_guess],
                bounds=[[A_bounds[0], t0_bounds[0], gamma_bounds[0], f_bounds[0], trise_bounds[0], tfall_bounds[0]],
                        [A_bounds[1], t0_bounds[1], gamma_bounds[1], f_bounds[1], trise_bounds[1], tfall_bounds[1]]],
                ftol=A_guess / 20.)
        except (ValueError, RuntimeError, OptimizeWarning):
            try:
                pout, pcov = curve_fit(
                    self.model,
                    times,
                    fluxpsf,
                    p0=[A_guess, t0_guess, gamma_guess, f_guess, trise_guess, tfall_guess],
                    bounds=[[A_bounds[0], t0_bounds[0], gamma_bounds[0], f_bounds[0], trise_bounds[0], tfall_bounds[0]],
                            [A_bounds[1], t0_bounds[1], gamma_bounds[1], f_bounds[1], trise_bounds[1],
                             tfall_bounds[1]]],
                    ftol=A_guess / 3.)
            except (ValueError, RuntimeError, OptimizeWarning):
                pout = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

        self.parameters = pout
        predictions = self.model(
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


def mag_to_flux(mag):
    """Converts a list of magnitudes into flux."""
    return 10 ** (-(mag + 48.6) / 2.5 + 26.0)


class SNParametricModelExtractor(FeatureExtractorSingleBand):
    """Fits a SNe parametric model to the light curve and provides
    the fitted parameters as features."""

    def __init__(self):
        self.sn_model = SNModelScipy()

    def get_features_keys(self) -> List[str]:
        return [
            'sn_model_a',
            'sn_model_t0',
            'sn_model_f',
            'sn_model_gamma',
            'sn_model_t_rise',
            'sn_model_t_fall',
            'sn_model_chi2_per_degree'
        ]

    def get_required_keys(self) -> List[str]:
        return ['mjd', 'magpsf', 'sigmapsf', 'fid']

    def compute_feature_in_one_band(self, detections, band=None, **kwargs):
        oids = detections.index.unique()
        sn_params = []

        detections = detections.sort_values('mjd')
        columns = self.get_features_keys_with_band(band)

        for oid in oids:
            oid_detections = detections.loc[[oid]]
            if band not in oid_detections.fid.values:
                logging.info(
                    f'extractor=SN parametric model object={oid} required_cols={self.get_required_keys()} band={band}')
                nan_df = self.nan_df(oid)
                nan_df.columns = columns
                sn_params.append(nan_df)
                continue

            oid_band_detections = oid_detections[oid_detections.fid == band]

            oid_band_detections = oid_band_detections[['mjd', 'magpsf', 'sigmapsf']]
            oid_band_detections = oid_band_detections.dropna()

            times = oid_band_detections['mjd'].values
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

            data = np.array(model_parameters).reshape([1, len(self.get_features_keys())])
            df = pd.DataFrame(
                data=data,
                columns=self.get_features_keys_with_band(band),
                index=[oid]
            )
            sn_params.append(df)
        sn_params = pd.concat(sn_params, axis=0)
        sn_params.index.name = 'oid'
        return sn_params
