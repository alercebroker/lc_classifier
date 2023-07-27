import numpy as np
from numba import jit
import math
from ..Base import Base
from scipy.optimize import minimize, minimize_scalar


@jit(nopython=True)
def iar_phi_kalman_numba(x, t, y, yerr, standarized):
    n = len(y)
    Sighat = 1.0
    if not standarized:
        Sighat = np.var(y) * Sighat
    xhat = np.zeros(shape=(1, n))
    delta = np.diff(t)
    Q = Sighat
    phi = x
    G = 1.0
    sum_Lambda = 0.0
    sum_error = 0.0
    if np.isnan(phi):
        phi = 1.1
    if abs(phi) < 1:
        for i in range(n - 1):
            Lambda = G * Sighat * G + yerr[i + 1] ** 2
            if (Lambda <= 0) or np.isnan(Lambda):
                sum_Lambda = float(n * 1e10)
                break
            phi2 = phi ** delta[i]
            F = phi2
            phi2 = 1 - phi ** (delta[i] * 2)
            Qt = phi2 * Q
            sum_Lambda = sum_Lambda + np.log(Lambda)
            Theta = F * Sighat * G
            g_dot_xhat = G * xhat[0:1, i][0]
            sum_error = sum_error + (y[i] - g_dot_xhat) ** 2 / Lambda
            xhat[0:1, i + 1] = F * xhat[0:1, i] + Theta / Lambda * (y[i] - G * xhat[0:1, i])
            Sighat = F * Sighat * F + Qt - Theta / Lambda * Theta
        out = (sum_Lambda + sum_error) / n
        if np.isnan(sum_Lambda):
            out = 1e10
    else:
        out = 1e10
    return out


class IAR_phi(Base):
    """
    functions to compute an IAR model with Kalman filter.
    Author: Felipe Elorrieta.
    """
    def __init__(self, shared_data):
        super().__init__(shared_data)
        self.Data = ['magnitude', 'time', 'error']

    def IAR_phi_kalman(self, x, t, y, yerr, standarized=True, c=0.5):
        return iar_phi_kalman_numba(x, t, y, yerr, standarized)

    def fit(self, data):
        magnitude = data[0]
        time = data[1]
        error = data[2]

        if np.sum(error) == 0:
            error = np.zeros(len(magnitude))

        std = np.std(magnitude, ddof=1)
        ynorm = (magnitude-np.mean(magnitude)) / std
        deltanorm = error / std

        out = minimize_scalar(
            self.IAR_phi_kalman,
            args=(time, ynorm, deltanorm),
            bounds=(0, 1),
            method="bounded",
            options={'xatol': 1e-12, 'maxiter': 50000})

        phi = out.x
        try:
            phi = phi[0][0]
        except:
            phi = phi

        return phi


class CIAR_phiR_beta(Base):
    """
    functions to compute an IAR model with Kalman filter.
    Author: Felipe Elorrieta.
    (beta version)
    """
    def __init__(self, shared_data):
        super().__init__(shared_data)
        self.Data = ['magnitude', 'time', 'error']

    def CIAR_phi_kalman(self, x, t, y, yerr, mean_zero=True, standarized=True, c=0.5):
        n = len(y)
        Sighat = np.zeros(shape=(2, 2))
        Sighat[0, 0] = 1
        Sighat[1, 1] = c
        if not standarized:
            Sighat = np.var(y)*Sighat
        if not mean_zero:
            y = y-np.mean(y)
        xhat = np.zeros(shape=(2, n))
        delta = np.diff(t)
        Q = Sighat
        phi_R = x[0]
        phi_I = x[1]
        F = np.zeros(shape=(2, 2))
        G = np.zeros(shape=(1, 2))
        G[0, 0] = 1
        phi = complex(phi_R, phi_I)
        Phi = abs(phi)
        psi = np.arccos(phi_R/Phi)
        sum_Lambda = 0
        sum_error = 0
        if np.isnan(phi):
            phi = 1.1
        if abs(phi) < 1:
            for i in range(n-1):
                Lambda = np.dot(np.dot(G, Sighat), G.transpose())+yerr[i+1]**2
                if (Lambda <= 0) or np.isnan(Lambda):
                    sum_Lambda = n*1e10
                    break
                phi2_R = (Phi**delta[i])*np.cos(delta[i]*psi)
                phi2_I = (Phi**delta[i])*np.sin(delta[i]*psi)
                phi2 = 1-abs(phi**delta[i])**2
                F[0, 0] = phi2_R
                F[0, 1] = -phi2_I
                F[1, 0] = phi2_I
                F[1, 1] = phi2_R
                Qt = phi2*Q
                sum_Lambda = sum_Lambda+np.log(Lambda)
                Theta = np.dot(np.dot(F, Sighat), G.transpose())
                sum_error = sum_error + (y[i]-np.dot(G, xhat[0:2, i]))**2/Lambda
                xhat[0:2, i+1] = (
                        np.dot(F, xhat[0:2,i])
                        + np.dot(np.dot(Theta, np.linalg.inv(Lambda)), (y[i]-np.dot(G, xhat[0:2, i]))))
                Sighat = (
                        np.dot(np.dot(F, Sighat), F.transpose())
                        + Qt
                        - np.dot(np.dot(Theta, np.linalg.inv(Lambda)), Theta.transpose()))
            out = (sum_Lambda + sum_error)/n
            if np.isnan(sum_Lambda):
                out = 1e10
        else:
            out = 1e10
        return out

    def fit(self, data):
        magnitude = data[0]
        time = data[1]
        error = data[2]

        niter = 4
        seed = 1234

        ynorm = (magnitude-np.mean(magnitude))/np.sqrt(np.var(magnitude, ddof=1))
        deltanorm = error/np.sqrt(np.var(magnitude, ddof=1))

        np.random.seed(seed)
        aux = 1e10
        br = 0
        if np.sum(error) == 0:
            deltanorm = np.zeros(len(error))
        for i in range(niter):
            phi_R = 2*np.random.uniform(0, 1, 1)-1
            phi_I = 2*np.random.uniform(0, 1, 1)-1
            bnds = ((-0.9999, 0.9999), (-0.9999, 0.9999))
            out = minimize(
                self.CIAR_phi_kalman,
                np.array([phi_R, phi_I]),
                args=(time, ynorm, deltanorm),
                bounds=bnds,
                method='L-BFGS-B')
            value = out.fun
            if aux > value:
                par = out.x
                aux = value
                br = br+1
            if aux <= value and br > 1 and i > math.trunc(niter/2):
                break
        if aux == 1e10:
            par = np.zeros(2)
        return par[0]
