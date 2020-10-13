import os
import numpy as np
import tensorflow as tf


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

    def fit(self, times, targets, errors=None):
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
