import os
from typing import List

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import tensorflow as tf
from ..core.base import FeatureExtractorSingleBand
from scipy.optimize import OptimizeWarning
import logging


os.environ["CUDA_VISIBLE_DEVICES"] = ""


class SNModel(object):
    def __init__(self):
        self.A_param = tf.Variable(2.0)
        self.t0 = tf.Variable(5.0)
        self.f_param = tf.Variable(0.0)
        self.gamma_param = tf.Variable(20.0)
        self.t_rise_param = tf.Variable(7.0)
        self.t_fall_param = tf.Variable(20.0)

        self.beta = 1.0 / 3.0
        self.l2_factor = 0.05

        self.optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=0.01)

    def __call__(self, times):
        A = tf.abs(self.A_param)
        t0 = tf.abs(self.t0)
        f = tf.sigmoid(self.f_param)
        gamma = tf.abs(50 * self.gamma_param) + 1.0
        t_rise = tf.abs(self.t_rise_param) + 1.0
        t_fall = tf.abs(self.t_fall_param) + 1.0

        t1 = t0 + gamma
        sigmoid = tf.sigmoid(self.beta * (times - t1))
        den = 1 + tf.exp(-(times - t0) / t_rise)
        flux = (A * (1 - f) * tf.exp(-(times - t1) / t_fall) / den
                * sigmoid
                + A * (1. - f * (times - t0) / gamma) / den
                * (1 - sigmoid))
        return flux

    def set_guess(self, times, targets):
        self.A_param.assign(np.max(targets) * 2)
        self.t0.assign(times[np.argmax(targets)])
        self.f_param.assign(0.0)
        self.gamma_param.assign(0.1)
        self.t_rise_param.assign(5.0)
        self.t_fall_param.assign(20.0)

    def fit(self, times, targets):
        self.set_guess(times, targets)
        for iteration in range(250):
            loss_value, gradient_value = self._gradient(times, targets)
            self.optimizer.apply_gradients(
                zip(gradient_value, self.get_trainable_variables()))
        return loss_value.numpy()

    def _loss(self, predictions, targets):
        return (tf.reduce_mean((predictions - targets) ** 2) / tf.reduce_mean(targets ** 2)
                + (tf.nn.relu(self.gamma_param - 50)
                   + tf.nn.relu(self.t_rise_param - 100)
                   + tf.nn.relu(self.t_fall_param - 100)) * self.l2_factor
                )

    def _gradient(self, times, targets):
        with tf.GradientTape() as tape:
            loss_value = self._loss(self.__call__(times), targets)
        return loss_value, tape.gradient(
            loss_value,
            self.get_trainable_variables())

    def get_trainable_variables(self):
        return (self.A_param,
                self.t0,
                self.f_param,
                self.gamma_param,
                self.t_rise_param,
                self.t_fall_param)

    def get_model_parameters(self):
        A = tf.abs(self.A_param)
        t0 = tf.abs(self.t0)
        f = tf.sigmoid(self.f_param)
        gamma = tf.abs(50 * self.gamma_param) + 1.0
        t_rise = tf.abs(self.t_rise_param) + 1.0
        t_fall = tf.abs(self.t_fall_param) + 1.0
        return [
            A.numpy(),
            t0.numpy(),
            f.numpy(),
            gamma.numpy(),
            t_rise.numpy(),
            t_fall.numpy()]


class SNModelScipy(object):
    def __init__(self):
        self.beta = 1.0 / 3.0
        self.parameters = None

    def model(self, times, A, t0, gamma, f, t_rise, t_fall):
        t1 = t0 + gamma

        sigmoid = 1.0 / (1.0 + np.exp(-self.beta * (times - t1)))
        den = 1 + tf.exp(-(times - t0) / t_rise)
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
    return 10 ** (-(mag + 48.6) / 2.5 + 26.0)


class SNParametricModelExtractor(FeatureExtractorSingleBand):
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
            oid_detections = detections.loc[oid]
            if band not in oid_detections.fid.values:
                logging.warning(
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
