import numpy as np
from numba import jit
from ..Base import Base


@jit(nopython=True)
def SFarray(jd, mag, err):
    """
    Calculates an array with (m(ti)-m(tj)), with (err(t)^2+err(t+tau)^2) and
    another with tau=dt

    inputs:
    jd: julian days array
    mag: magnitudes array
    err: error of magnitudes array

    outputs:
    tauarray: array with the difference in time (ti-tj)
    sfarray: array with |m(ti)-m(tj)|
    errarray: array with err(ti)^2+err(tj)^2
    """
    sfarray = []
    tauarray = []
    errarray = []
    err_squared = err**2
    len_mag = len(mag)
    for i in range(len_mag):
        for j in range(i+1, len_mag):
            dm = mag[i] - mag[j]
            sigma = err_squared[i] + err_squared[j]
            dt = jd[j] - jd[i]
            sfarray.append(np.abs(dm))
            tauarray.append(dt)
            errarray.append(sigma)
    sfarray = np.array(sfarray)
    tauarray = np.array(tauarray)
    errarray = np.array(errarray)
    return tauarray, sfarray, errarray


class SF_ML_amplitude(Base):
    """
    Fit the model A*tau^gamma to the SF, finding the maximum value of the likelihood.
    Based on Schmidt et al. 2010.
    """
    def __init__(self, shared_data):
        super().__init__(shared_data)
        self.Data = ['magnitude', 'time', 'error']

    def bincalc(self, nbin=0.1, bmin=5, bmax=2000):
        """
        calculate the bin range, in logscale

        inputs:
        nbin: size of the bin in log scale
        bmin: minimum value of the bins
        bmax: maximum value of the bins

        output: bins array
        """
        logbmin = np.log10(bmin)
        logbmax = np.log10(bmax)

        logbins = np.arange(logbmin, logbmax, nbin)

        bins = 10**logbins
        return bins

    def sf_formula(self, jd, mag, errmag, nbin=0.1, bmin=5, bmax=2000):
        dtarray, dmagarray, sigmaarray = SFarray(jd, mag, errmag)
        ndt = np.where((dtarray <= 365) & (dtarray >= 5))
        dtarray = dtarray[ndt]
        dmagarray = dmagarray[ndt]
        sigmaarray = sigmaarray[ndt]

        bins = self.bincalc(nbin, bmin, bmax)

        sf_list = []
        tau_list = []

        for i in range(0, len(bins)-1):
            n = np.where((dtarray >= bins[i]) & (dtarray < bins[i+1]))
            nobjbin = len(n[0])
            if nobjbin >= 1:
                dmag1 = (dmagarray[n])**2
                derr1 = (sigmaarray[n])
                sf = (dmag1-derr1)
                mean_sf = np.mean(sf)
                if mean_sf < 0:
                    continue
                sff = np.sqrt(mean_sf)
                sf_list.append(sff)
                # central tau for the bin
                tau_list.append((bins[i]+bins[i+1])*0.5)

        SF = np.array(sf_list)
        tau = np.array(tau_list)

        if len(SF) < 2:
            tau = np.array([-99])
            SF = np.array([-99])

        return tau/365., SF

    def fit(self, data):
        mag = data[0]
        t = data[1]
        err = data[2]

        tau, sf = self.sf_formula(t, mag, err)

        if sf[0] == -99:
            a = -0.5
            gamma = -0.5

        else:
            y = np.log10(sf)
            x = np.log10(tau)
            x = x[np.where((tau <= 0.5) & (tau > 0.01))]
            y = y[np.where((tau <= 0.5) & (tau > 0.01))]
            try:
                coefficients = np.polyfit(x, y, 1)
                a = 10**(coefficients[1])
                gamma = coefficients[0]

                if a < 0.005:
                    a = 0.0
                    gamma = 0.0
                elif a > 15:
                    a = 15

                gamma = np.clip(gamma, -0.5, 3.0)

            except:
                a = -0.5
                gamma = -0.5

        self.shared_data['g_sf'] = gamma
        return a


class SF_ML_gamma(Base):
    def __init__(self, shared_data):
        super().__init__(shared_data)
        self.Data = ['magnitude', 'time', 'error']

    def fit(self, data):
        try:
            g_sf = self.shared_data['g_sf']
            return g_sf
        except KeyError:
            raise Exception("Please run SF_amplitude first to generate values for SF_gamma")
