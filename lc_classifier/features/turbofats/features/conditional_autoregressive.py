from ..Base import Base
from numba import jit
import numpy as np
from scipy.optimize import minimize
import logging


@jit(nopython=True)
def car_likelihood(parameters, t, x, error_vars):
    sigma = parameters[0]
    tau = parameters[1]

    b = np.mean(x) / tau
    epsilon = 1e-300
    cte_neg = -np.infty
    num_datos = len(x)

    Omega = []
    x_hat = []
    a = []
    x_ast = []

    Omega.append(np.array([tau * (sigma ** 2) / 2.0]))
    x_hat.append(np.array([0.0]))
    a.append(np.array([0.0]))
    x_ast.append(x[0] - b * tau)

    loglik = np.array([0.0])

    for i in range(1, num_datos):
        a_new = np.exp(-(t[i] - t[i - 1]) / tau)
        x_ast.append(x[i] - b * tau)
        x_hat_val = (
            a_new * x_hat[i - 1] +
            (a_new * Omega[i - 1] / (Omega[i - 1] + error_vars[i - 1])) *
            (x_ast[i - 1] - x_hat[i - 1]))
        x_hat.append(x_hat_val)

        Omega.append(
            Omega[0] * (1 - (a_new ** 2)) + (a_new ** 2) * Omega[i - 1] *
            (1 - (Omega[i - 1] / (Omega[i - 1] + error_vars[i - 1]))))

        loglik_inter = np.log(
            ((2 * np.pi * (Omega[i] + error_vars[i])) ** -0.5) *
            (np.exp(-0.5 * (((x_hat[i] - x_ast[i]) ** 2) /
            (Omega[i] + error_vars[i]))) + epsilon))

        loglik = loglik + loglik_inter

        if loglik[0] <= cte_neg:
            print('CAR lik --> inf')
            return None

    # the minus one is to perform maximization using the minimize function
    return -loglik


class CAR_sigma(Base):
    def __init__(self, shared_data):
        super().__init__(shared_data)
        self.Data = ['magnitude', 'time', 'error']

    def CAR_Lik(self, parameters, t, x, error_vars):
        return car_likelihood(parameters, t, x, error_vars)

    def calculateCAR(self, time, data, error):
        x0 = np.array([0.01, 100])
        bnds = ((0, 20), (0.000001, 2000))

        try:
            res = minimize(self.CAR_Lik, x0, args=(time, data, error),
                           method='L-BFGS-B', bounds=bnds)
            sigma = res.x[0]
            tau = res.x[1]
        except TypeError:
            logging.warning('CAR optimizer raised an exception')
            sigma = np.nan
            tau = np.nan
        self.shared_data['tau'] = tau
        return sigma

    def fit(self, data):
        N = len(data[0])
        magnitude = data[0].reshape((N, 1))
        time = data[1].reshape((N, 1))
        error = data[2].reshape((N, 1)) ** 2

        a = self.calculateCAR(time, magnitude, error)

        return a


class CAR_tau(Base):
    def __init__(self, shared_data):
        super().__init__(shared_data)
        self.Data = ['magnitude', 'time', 'error']

    def fit(self, data):
        try:
            tau = self.shared_data['tau']
            return tau
        except:
            print("error: please run CAR_sigma first to generate values for CAR_tau")


class CAR_mean(Base):
    def __init__(self, shared_data):
        super().__init__(shared_data)
        self.Data = ['magnitude', 'time', 'error']

    def fit(self, data):
        magnitude = data[0]
        try:
            tau = self.shared_data['tau']
            return np.mean(magnitude) / tau
        except:
            print("error: please run CAR_sigma first to generate values for CAR_mean")
