import os
import numpy as np
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf

from late_classifier.features.core.base import FeatureExtractorSingleBand


class SNModel(object):
    def __init__(self):
        self.A_param = tf.Variable(2.0)
        self.t0 = tf.Variable(5.0)
        self.f_param = tf.Variable(0.0)
        self.gamma_param = tf.Variable(20.0)
        self.t_rise_param = tf.Variable(7.0)
        self.t_fall_param = tf.Variable(20.0)

        self.beta = 1.0/3.0
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
        den = 1 + tf.exp(-(times - t0)/t_rise)
        flux = (A * (1 - f) * tf.exp(-(times - t1) / t_fall) / den
                * sigmoid
                + A * (1. - f * (times - t0) / gamma) / den
                *(1-sigmoid))
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
        return (tf.reduce_mean((predictions - targets)**2) / tf.reduce_mean(targets ** 2)
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


# class SNModelScipy(FeatureExtractorSingleBand):
#     def __init__(self):
#         pass
# 
#     def fit(self, times, targets):
# 
# 
# def doSN(SN, doplot=False, verbose=False):
#     fluxpsf = targets
#     mjds = times
#         
#     A_bounds = [max(fluxpsf) / 3., max(fluxpsf) * 3.]
#     t0_bounds = [0, 50]
#     gamma_bounds = [1., 60.]
#     f_bounds = [0, 1.]
#     trise_bounds = [1., 100.]
#     tfall_bounds = [1., 100.]
#             
#     A_guess = max(A_bounds[1], min(A_bounds[0], max(fluxpsf)))
#     t0_guess = 0 #min(t0_bounds[1], max(t0_bounds[0], times[np.argmax(fluxpsf)]))
#     mask = fluxpsf >= np.percentile(fluxpsf, 33)
#     if mask.sum() > 0:
#         gamma_guess = min(gamma_bounds[1], max(gamma_bounds[0], times[mask].max() - times[mask].min()))
#     else:
#         gamma_guess = 2.
#     f_guess = 0.5
#     trise_guess = min(trise_bounds[1], max(trise_bounds[0], t0_guess - times.min()))
#     tfall_guess = 20.
#     
#     # reference guess
#     p0 = [A_guess, t0_guess, gamma_guess, f_guess, trise_guess, tfall_guess]
#     
#     if mask_fid.sum() <= 4:
#         sol[fid] = [A_guess, t0_guess, gamma_guess, f_guess, trise_guess, tfall_guess]
#     else:
#         # update times and flux
#         times = mjds[mask_fid] - min(mjds)
#         fluxpsf_fid = fluxpsf[mask_fid]
#         
#         if doplot:
#             ax.scatter(times, fluxpsf_fid, color=colors[fid])
#             
#         if verbose:
#             print(A_guess, t0_guess, gamma_guess, f_guess, trise_guess, tfall_guess)
# 
#         # get parameters
#         try:
#             if verbose:
#                 t1 = time.time()
#             pout, pcov = scipy.optimize.curve_fit(
#                 func,
#                 times,
#                 fluxpsf_fid,
#                 p0=[A_guess, t0_guess, gamma_guess, f_guess, trise_guess, tfall_guess],
#                 bounds=[[A_bounds[0], t0_bounds[0], gamma_bounds[0], f_bounds[0], trise_bounds[0], tfall_bounds[0]],
#                         [A_bounds[1], t0_bounds[1], gamma_bounds[1], f_bounds[1], trise_bounds[1], tfall_bounds[1]]],
#                 ftol=A_guess / 20.)
#             if verbose:
#                 print("curve fit time:", time.time() - t1)
#             except:
#                 try:
#                     print("\nTrying different guess...")
#                     A_guess, t0_guess, gamma_guess, f_guess, trise_guess, tfall_guess = p0
#                     pout, pcov = scipy.optimize.curve_fit(
#                         func,
#                         times,
#                         fluxpsf_fid,
#                         p0=[A_guess, t0_guess, gamma_guess, f_guess, trise_guess, tfall_guess],
#                         bounds=[[A_bounds[0], t0_bounds[0], gamma_bounds[0], f_bounds[0], trise_bounds[0], tfall_bounds[0]],
#                                 [A_bounds[1], t0_bounds[1], gamma_bounds[1], f_bounds[1], trise_bounds[1], tfall_bounds[1]]],
#                         ftol=A_guess / 3.)
#                 except:
#                     print("Failed convergence...")
#                     pout = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
# 
#             # save parameters
#             sol[fid] = pout
# 
#             # update guess to be used in next band
#             A_guess, t0_guess, gamma_guess, f_guess, trise_guess, tfall_guess = sol[fid]
#             
#             if doplot:
#                 timesint = np.linspace(0, mjds.max() - mjds.min(), 100)
#                 ax.plot(timesint, func(timesint, *pout, True), color=colors[fid])
#                 plt.savefig("SN/%s_fit.png" % SN)
# 
#     # return next solution
#     return sol
# 
# 
#     def get_model_parameters(self):
#         pass


class SNParametricModelExtractor(FeatureExtractorSingleBand):
    def __init__(self):
        super().__init__()
        self.features_keys = [
            'sn_model_a',
            'sn_model_t0',
            'sn_model_f',
            'sn_model_gamma',
            'sn_model_t_rise',
            'sn_model_t_fall',
            'sn_model_fit_error'
        ]
        self.sn_model = SNModel()

    def _compute_features(self, detections, **kwargs):
        if len(detections) > 0:
            detections = detections[['mjd', 'magpsf_corr']]
            detections = detections.dropna()

            times = detections['mjd'].values
            times = times - np.min(times)
            targets = detections['magpsf_corr'].values
            targets = self._mag_to_flux(targets)

            times = times.astype(np.float32)
            targets = targets.astype(np.float32)
            
            fit_error = self.sn_model.fit(times, targets)
            model_parameters = self.sn_model.get_model_parameters()
            model_parameters.append(fit_error)

            data = np.array(model_parameters).reshape([1, len(self.features_keys)])
            df = pd.DataFrame(
                data=data,
                columns=self.features_keys,
                index=[detections.index.values[0]]
            )
            df.index.name = 'oid'
            return df
        else:
            return pd.DataFrame()

    def _mag_to_flux(self, mag):
        return 10**(-(mag + 48.6) / 2.5 + 26.0)

