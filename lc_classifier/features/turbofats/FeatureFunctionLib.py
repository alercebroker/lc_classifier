import math

import warnings

import numpy as np

from scipy import stats
from statsmodels.tsa import stattools
from scipy.interpolate import interp1d
from scipy.stats import chi2

from .Base import Base

from .features.irregular_autoregressive import IAR_phi, CIAR_phiR_beta
from .features.structure_function import SF_ML_amplitude, SF_ML_gamma
from .features.conditional_autoregressive import CAR_mean, CAR_sigma, CAR_tau


class Amplitude(Base):
    """Half the difference between the maximum and the minimum magnitude"""

    def __init__(self, shared_data):
        super().__init__(shared_data)
        self.Data = ['magnitude']

    def fit(self, data):
        magnitude = data[0]
        n = len(magnitude)
        sorted_mag = np.sort(magnitude)

        return (np.median(sorted_mag[int(-math.ceil(0.05 * n)):]) -
                np.median(sorted_mag[0:int(math.ceil(0.05 * n))])) / 2.0


class Rcs(Base):
    """Range of cumulative sum"""

    def __init__(self, shared_data):
        super().__init__(shared_data)
        self.Data = ['magnitude']

    def fit(self, data):
        magnitude = data[0]
        sigma = np.std(magnitude)
        N = len(magnitude)
        m = np.mean(magnitude)
        s = np.cumsum(magnitude - m) * 1.0 / (N * sigma)
        R = np.max(s) - np.min(s)
        return R


class StetsonK(Base):
    def __init__(self, shared_data):
        super().__init__(shared_data)
        self.Data = ['magnitude', 'error']

    def fit(self, data):
        magnitude = data[0]
        error = data[2]
        n = len(magnitude)

        mean_mag = (np.sum(magnitude/(error*error))/np.sum(1.0 / (error * error)))
        sigmap = (np.sqrt(n * 1.0 / (n - 1)) *
                  (magnitude - mean_mag) / error)

        k = (1 / np.sqrt(n * 1.0) * np.sum(np.abs(sigmap)) / np.sqrt(np.sum(sigmap ** 2)))
        return k


class Meanvariance(Base):
    """variability index"""
    def __init__(self, shared_data):
        super().__init__(shared_data)
        self.Data = ['magnitude']

    def fit(self, data):
        magnitude = data[0]
        return np.std(magnitude) / np.mean(magnitude)


class Autocor_length(Base):
    def __init__(self, shared_data, lags=100):
        super().__init__(shared_data)
        self.Data = ['magnitude']
        self.nlags = lags

    def fit(self, data):

        magnitude = data[0]
        ac = stattools.acf(magnitude, nlags=self.nlags, fft=False)

        k = next((index for index, value in
                 enumerate(ac) if value < np.exp(-1)), None)

        while k is None:
            if self.nlags > len(magnitude):
                warnings.warn('Setting autocorrelation length as light curve length')
                return len(magnitude)
            self.nlags = self.nlags + 100
            ac = stattools.acf(magnitude, nlags=self.nlags, fft=False)
            k = next((index for index, value in
                      enumerate(ac) if value < np.exp(-1)), None)

        return k


class SlottedA_length(Base):
    def __init__(self, shared_data, T=-99):
        """
        lc: MACHO lightcurve in a pandas DataFrame
        k: lag (default: 1)
        T: tau (slot size in days. default: 4)
        """
        super().__init__(shared_data)
        self.Data = ['magnitude', 'time']

        SlottedA_length.SAC = []

        self.T = T

    def slotted_autocorrelation(self, data, time, T, K,
                                second_round=False, K1=100):

        slots = np.zeros((K, 1))
        i = 1

        # make time start from 0
        time = time - np.min(time)

        # subtract mean from mag values
        m = np.mean(data)
        data = data - m

        prod = np.zeros((K, 1))
        pairs = np.subtract.outer(time, time)
        pairs[np.tril_indices_from(pairs)] = 10000000

        ks = np.int64(np.floor(np.abs(pairs) / T + 0.5))

        # We calculate the slotted autocorrelation for k=0 separately
        idx = np.where(ks == 0)
        prod[0] = ((sum(data ** 2) + sum(data[idx[0]] *
                   data[idx[1]])) / (len(idx[0]) + len(data)))
        slots[0] = 0

        # We calculate it for the rest of the ks
        if second_round is False:
            for k in np.arange(1, K):
                idx = np.where(ks == k)
                if len(idx[0]) != 0:
                    prod[k] = sum(data[idx[0]] * data[idx[1]]) / (len(idx[0]))
                    slots[i] = k
                    i = i + 1
                else:
                    prod[k] = np.infty
        else:
            for k in np.arange(K1, K):
                idx = np.where(ks == k)
                if len(idx[0]) != 0:
                    prod[k] = sum(data[idx[0]] * data[idx[1]]) / (len(idx[0]))
                    slots[i - 1] = k
                    i = i + 1
                else:
                    prod[k] = np.infty
            np.trim_zeros(prod, trim='b')

        slots = np.trim_zeros(slots, trim='b')
        return prod / prod[0], np.int64(slots).flatten()

    def fit(self, data):
        magnitude = data[0]
        time = data[1]
        N = len(time)

        if self.T == -99:
            deltaT = time[1:] - time[:-1]
            sorted_deltaT = np.sort(deltaT)
            self.T = sorted_deltaT[int(N * 0.05)+1]

        K = 100

        [SAC, slots] = self.slotted_autocorrelation(magnitude, time, self.T, K)

        SAC2 = SAC[slots]
        SlottedA_length.autocor_vector = SAC2

        k = next((index for index, value in
                 enumerate(SAC2) if value < np.exp(-1)), None)

        while k is None:
            K = K+K

            if K > (np.max(time) - np.min(time)) / self.T:
                break
            else:
                [SAC, slots] = self.slotted_autocorrelation(magnitude,
                                                            time, self.T, K,
                                                            second_round=True,
                                                            K1=K/2)
                SAC2 = SAC[slots]
                k = next((index for index, value in
                         enumerate(SAC2) if value < np.exp(-1)), None)

        return slots[k] * self.T

    def getAtt(self):
        return SlottedA_length.autocor_vector


class StetsonK_AC(SlottedA_length):
    def __init__(self, shared_data):
        super().__init__(shared_data)
        self.Data = ['magnitude', 'time', 'error']

    def fit(self, data):
        try:
            a = StetsonK_AC(self.shared_data)
            autocor_vector = a.getAtt()
            N_autocor = len(autocor_vector)
            sigmap = (np.sqrt(N_autocor * 1.0 / (N_autocor - 1)) *
                      (autocor_vector - np.mean(autocor_vector)) /
                      np.std(autocor_vector))

            K = (1 / np.sqrt(N_autocor * 1.0) *
                 np.sum(np.abs(sigmap)) / np.sqrt(np.sum(sigmap ** 2)))

            return K
        except:
            print("error: please run SlottedA_length first to generate values for StetsonK_AC ")


class Con(Base):
    """Index introduced for selection of variable starts from OGLE database.


    To calculate Con, we counted the number of three consecutive measurements
    that are out of 2sigma range, and normalized by N-2
    Pavlos not happy
    """
    def __init__(self, shared_data, consecutiveStar=3):
        super().__init__(shared_data)
        self.Data = ['magnitude']
        self.consecutiveStar = consecutiveStar

    def fit(self, data):
        magnitude = data[0]
        N = len(magnitude)
        if N < self.consecutiveStar:
            return 0
        sigma = np.std(magnitude)
        m = np.mean(magnitude)
        count = 0

        for i in range(N - self.consecutiveStar + 1):
            flag = 0
            for j in range(self.consecutiveStar):
                if (magnitude[i + j] > m + 2 * sigma
                        or magnitude[i + j] < m - 2 * sigma):
                    flag = 1
                else:
                    flag = 0
                    break
            if flag:
                count = count + 1
        return count * 1.0 / (N - self.consecutiveStar + 1)


class Color(Base):
    """Average color for each MACHO lightcurve
    mean(B1) - mean(B2)
    """
    def __init__(self, shared_data):
        super().__init__(shared_data)
        self.Data = ['magnitude', 'time', 'magnitude2']

    def fit(self, data):
        magnitude = data[0]
        magnitude2 = data[3]
        return np.mean(magnitude) - np.mean(magnitude2)


class Beyond1Std(Base):
    """Percentage of points beyond one st. dev. from the weighted
    (by photometric errors) mean
    """
    def __init__(self, shared_data):
        super().__init__(shared_data)
        self.Data = ['magnitude', 'error']

    def fit(self, data):
        magnitude = data[0]
        error = data[2]
        n = len(magnitude)

        weighted_mean = np.average(magnitude, weights=1 / error ** 2)

        # Standard deviation with respect to the weighted mean
        var = sum((magnitude - weighted_mean) ** 2)
        std = np.sqrt((1.0 / (n - 1)) * var)

        count = np.sum(np.logical_or(magnitude > weighted_mean + std,
                                     magnitude < weighted_mean - std))
        return float(count) / n


class SmallKurtosis(Base):
    """Small sample kurtosis of the magnitudes.

    See http://www.xycoon.com/peakedness_small_sample_test_1.htm
    """
    def __init__(self, shared_data):
        super().__init__(shared_data)
        self.Data = ['magnitude']

    def fit(self, data):
        magnitude = data[0]
        n = len(magnitude)
        mean = np.mean(magnitude)
        std = np.std(magnitude)

        S = sum(((magnitude - mean) / std) ** 4)

        c1 = float(n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))
        c2 = float(3 * (n - 1) ** 2) / ((n - 2) * (n - 3))

        return c1 * S - c2


class Std(Base):
    """Standard deviation of the magnitudes"""
    def __init__(self, shared_data):
        super().__init__(shared_data)
        self.Data = ['magnitude']

    def fit(self, data):
        magnitude = data[0]
        return np.std(magnitude)


class Skew(Base):
    """Skewness of the magnitudes"""
    def __init__(self, shared_data):
        super().__init__(shared_data)
        self.Data = ['magnitude']

    def fit(self, data):
        magnitude = data[0]
        return stats.skew(magnitude)


class MaxSlope(Base):
    """
    Examining successive (time-sorted) magnitudes, the maximal first difference
    (value of delta magnitude over delta time)
    """
    def __init__(self, shared_data):
        super().__init__(shared_data)
        self.Data = ['magnitude', 'time']

    def fit(self, data):

        magnitude = data[0]
        time = data[1]
        slope = np.abs(magnitude[1:] - magnitude[:-1]) / (time[1:] - time[:-1])
        np.max(slope)

        return np.max(slope)


class MedianAbsDev(Base):
    def __init__(self, shared_data):
        super().__init__(shared_data)
        self.Data = ['magnitude']

    def fit(self, data):
        magnitude = data[0]
        median = np.median(magnitude)

        devs = (abs(magnitude - median))

        return np.median(devs)


class MedianBRP(Base):
    """Median buffer range percentage

    Fraction (<= 1) of photometric points within amplitude/10
    of the median magnitude
    """
    def __init__(self, shared_data):
        super().__init__(shared_data)
        self.Data = ['magnitude']

    def fit(self, data):
        magnitude = data[0]
        median = np.median(magnitude)
        amplitude = (np.max(magnitude) - np.min(magnitude)) / 10
        n = len(magnitude)

        count = np.sum(np.logical_and(magnitude < median + amplitude,
                                      magnitude > median - amplitude))

        return float(count) / n


class PairSlopeTrend(Base):
    """
    Considering the last 30 (time-sorted) measurements of source magnitude,
    the fraction of increasing first differences minus the fraction of
    decreasing first differences.
    """
    def __init__(self, shared_data):
        super().__init__(shared_data)
        self.Data = ['magnitude']

    def fit(self, data):
        magnitude = data[0]
        data_last = magnitude[-30:]

        return (float(len(np.where(np.diff(data_last) > 0)[0]) -
                len(np.where(np.diff(data_last) <= 0)[0])) / 30)


class FluxPercentileRatioMid20(Base):
    def __init__(self, shared_data):
        super().__init__(shared_data)
        self.Data = ['magnitude']

    def fit(self, data):
        magnitude = data[0]
        sorted_data = np.sort(magnitude)
        lc_length = len(sorted_data)

        F_60_index = math.ceil(0.60 * lc_length)
        F_40_index = math.ceil(0.40 * lc_length)
        F_5_index = math.ceil(0.05 * lc_length)
        F_95_index = math.ceil(0.95 * lc_length)

        F_40_60 = sorted_data[F_60_index] - sorted_data[F_40_index]
        F_5_95 = sorted_data[F_95_index] - sorted_data[F_5_index]
        F_mid20 = F_40_60 / F_5_95

        return F_mid20


class FluxPercentileRatioMid35(Base):
    def __init__(self, shared_data):
        super().__init__(shared_data)
        self.Data = ['magnitude']

    def fit(self, data):
        magnitude = data[0]
        sorted_data = np.sort(magnitude)
        lc_length = len(sorted_data)

        F_325_index = math.ceil(0.325 * lc_length)
        F_675_index = math.ceil(0.675 * lc_length)
        F_5_index = math.ceil(0.05 * lc_length)
        F_95_index = math.ceil(0.95 * lc_length)

        F_325_675 = sorted_data[F_675_index] - sorted_data[F_325_index]
        F_5_95 = sorted_data[F_95_index] - sorted_data[F_5_index]
        F_mid35 = F_325_675 / F_5_95

        return F_mid35


class FluxPercentileRatioMid50(Base):
    def __init__(self, shared_data):
        super().__init__(shared_data)
        self.Data = ['magnitude']

    def fit(self, data):
        magnitude = data[0]
        sorted_data = np.sort(magnitude)
        lc_length = len(sorted_data)

        F_25_index = math.ceil(0.25 * lc_length)
        F_75_index = math.ceil(0.75 * lc_length)
        F_5_index = math.ceil(0.05 * lc_length)
        F_95_index = math.ceil(0.95 * lc_length)

        F_25_75 = sorted_data[F_75_index] - sorted_data[F_25_index]
        F_5_95 = sorted_data[F_95_index] - sorted_data[F_5_index]
        F_mid50 = F_25_75 / F_5_95

        return F_mid50


class FluxPercentileRatioMid65(Base):
    def __init__(self, shared_data):
        super().__init__(shared_data)
        self.Data = ['magnitude']

    def fit(self, data):
        magnitude = data[0]
        sorted_data = np.sort(magnitude)
        lc_length = len(sorted_data)

        F_175_index = math.ceil(0.175 * lc_length)
        F_825_index = math.ceil(0.825 * lc_length)
        F_5_index = math.ceil(0.05 * lc_length)
        F_95_index = math.ceil(0.95 * lc_length)

        F_175_825 = sorted_data[F_825_index] - sorted_data[F_175_index]
        F_5_95 = sorted_data[F_95_index] - sorted_data[F_5_index]
        F_mid65 = F_175_825 / F_5_95

        return F_mid65


class FluxPercentileRatioMid80(Base):
    def __init__(self, shared_data):
        super().__init__(shared_data)
        self.Data = ['magnitude']

    def fit(self, data):
        magnitude = data[0]
        sorted_data = np.sort(magnitude)
        lc_length = len(sorted_data)

        F_10_index = math.ceil(0.10 * lc_length)
        F_90_index = math.ceil(0.90 * lc_length)
        F_5_index = math.ceil(0.05 * lc_length)
        F_95_index = math.floor(0.95 * lc_length)

        F_10_90 = sorted_data[F_90_index] - sorted_data[F_10_index]
        F_5_95 = sorted_data[F_95_index] - sorted_data[F_5_index]
        F_mid80 = F_10_90 / F_5_95

        return F_mid80


class PercentDifferenceFluxPercentile(Base):
    def __init__(self, shared_data):
        super().__init__(shared_data)
        self.Data = ['magnitude']

    def fit(self, data):
        magnitude = data[0]
        median_data = np.median(magnitude)

        sorted_data = np.sort(magnitude)
        lc_length = len(sorted_data)
        F_5_index = math.ceil(0.05 * lc_length)
        F_95_index = math.ceil(0.95 * lc_length)
        F_5_95 = sorted_data[F_95_index] - sorted_data[F_5_index]

        percent_difference = F_5_95 / median_data

        return percent_difference


class PercentAmplitude(Base):
    def __init__(self, shared_data):
        super().__init__(shared_data)
        self.Data = ['magnitude']

    def fit(self, data):
        magnitude = data[0]
        median_data = np.median(magnitude)
        distance_median = np.abs(magnitude - median_data)
        max_distance = np.max(distance_median)

        percent_amplitude = max_distance / median_data

        return percent_amplitude


class LinearTrend(Base):
    def __init__(self, shared_data):
        super().__init__(shared_data)
        self.Data = ['magnitude', 'time']

    def fit(self, data):
        magnitude = data[0]
        time = data[1]
        regression_slope = stats.linregress(time, magnitude)[0]

        return regression_slope


class Eta_e(Base):
    def __init__(self, shared_data):
        super().__init__(shared_data)
        self.Data = ['magnitude', 'time']

    def fit(self, data):
        magnitude = data[0]
        time = data[1]
        w = 1.0 / np.power(np.subtract(time[1:], time[:-1]), 2)
        sigma2 = np.var(magnitude)

        S1 = np.sum(w * np.power(np.subtract(magnitude[1:], magnitude[:-1]), 2))
        S2 = np.sum(w)

        eta_e = (1 / sigma2) * (S1 / S2)
        return eta_e


class Mean(Base):
    def __init__(self, shared_data):
        super().__init__(shared_data)
        self.Data = ['magnitude']

    def fit(self, data):
        magnitude = data[0]
        B_mean = np.mean(magnitude)
        return B_mean


class Q31(Base):
    def __init__(self, shared_data):
        super().__init__(shared_data)

        self.Data = ['magnitude']

    def fit(self, data):
        magnitude = data[0]
        percentiles = np.percentile(magnitude, (25, 75))
        return percentiles[1] - percentiles[0]


class Q31_color(Base):
    def __init__(self, shared_data):
        super().__init__(shared_data)
        self.Data = ['magnitude', 'time', 'magnitude2']

    def fit(self, data):
        aligned_magnitude = data[4]
        aligned_magnitude2 = data[5]
        N = len(aligned_magnitude)
        b_r = aligned_magnitude[:N] - aligned_magnitude2[:N]
        percentiles = np.percentile(b_r, (25, 75))
        return percentiles[1] - percentiles[0]


class AndersonDarling(Base):
    def __init__(self, shared_data):
        super().__init__(shared_data)
        self.Data = ['magnitude']

    def fit(self, data):
        magnitude = data[0]
        ander = stats.anderson(magnitude)[0]
        return 1 / (1.0 + np.exp(-10 * (ander - 0.3)))


class Psi_CS_v2(Base):
    def __init__(self, shared_data):
        super().__init__(shared_data)
        self.Data = ['magnitude', 'time']

    def fit(self, data):
        try:
            magnitude = data[0]
            time = data[1]
            new_time_v2 = self.shared_data['new_time_v2']
            folded_data = magnitude[np.argsort(new_time_v2)]
            sigma = np.std(folded_data)
            N = len(folded_data)
            m = np.mean(folded_data)
            s = np.cumsum(folded_data - m) * 1.0 / (N * sigma)
            R = np.max(s) - np.min(s)

            return R
        except:
            print("error: please run PeriodLS_v2 first to generate values for Psi_CS_v2")


class Psi_eta_v2(Base):
    def __init__(self, shared_data):
        super().__init__(shared_data)
        self.Data = ['magnitude', 'time']

    def fit(self, data):
        try:
            magnitude = data[0]
            new_time_v2 = self.shared_data['new_time_v2']
            folded_data = magnitude[np.argsort(new_time_v2)]

            N = len(folded_data)
            sigma2 = np.var(folded_data)

            Psi_eta = (1.0 / ((N - 1) * sigma2) *
                       np.sum(np.power(folded_data[1:] - folded_data[:-1], 2)))

            return Psi_eta
        except:
            print("error: please run PeriodLS_v2 first to generate values for Psi_eta_v2")


class Gskew(Base):
    """Median-based measure of the skew"""
    def __init__(self, shared_data):
        super().__init__(shared_data)
        self.Data = ['magnitude']

    def fit(self, data):
        magnitude = np.array(data[0])
        median_mag = np.median(magnitude)
        F_3_value, F_97_value = np.percentile(magnitude, (3, 97))

        return (np.median(magnitude[magnitude <= F_3_value]) +
                np.median(magnitude[magnitude >= F_97_value])
                - 2*median_mag)


class StructureFunction_index_21(Base):
    def __init__(self, shared_data):
        super().__init__(shared_data)
        self.Data = ['magnitude', 'time']

    def fit(self, data):
        magnitude = data[0]
        time = data[1]

        Nsf = 100
        Np = 100
        sf1 = np.zeros(Nsf)
        sf2 = np.zeros(Nsf)
        sf3 = np.zeros(Nsf)
        f = interp1d(time, magnitude)

        time_int = np.linspace(np.min(time), np.max(time), Np)
        mag_int = f(time_int)

        for tau in np.arange(1, Nsf):
            sf1[tau-1] = np.mean(np.power(np.abs(mag_int[0:Np-tau] - mag_int[tau:Np]), 1.0))
            sf2[tau-1] = np.mean(np.abs(np.power(np.abs(mag_int[0:Np-tau] - mag_int[tau:Np]), 2.0)))
            sf3[tau-1] = np.mean(np.abs(np.power(np.abs(mag_int[0:Np-tau] - mag_int[tau:Np]), 3.0)))
        sf1_log = np.log10(np.trim_zeros(sf1))
        sf2_log = np.log10(np.trim_zeros(sf2))
        sf3_log = np.log10(np.trim_zeros(sf3))

        m_21, b_21 = np.polyfit(sf1_log, sf2_log, 1)
        m_31, b_31 = np.polyfit(sf1_log, sf3_log, 1)
        m_32, b_32 = np.polyfit(sf2_log, sf3_log, 1)
        self.shared_data['m_21'] = m_21
        self.shared_data['m_31'] = m_31
        self.shared_data['m_32'] = m_32

        return m_21


class StructureFunction_index_31(Base):
    def __init__(self, shared_data):
        super().__init__(shared_data)
        self.Data = ['magnitude', 'time']

    def fit(self, data):
        try:
            m_31 = self.shared_data['m_31']
            return m_31
        except:
            print("error: please run StructureFunction_index_21 first to generate values for all Structure Function")


class StructureFunction_index_32(Base):
    def __init__(self, shared_data):
        super().__init__(shared_data)
        self.Data = ['magnitude', 'time']

    def fit(self, data):
        try:
            m_32 = self.shared_data['m_32']
            return m_32
        except:
            print("error: please run StructureFunction_index_21 first to generate values for all Structure Function")


class Pvar(Base):
    """
    Calculate the probability of a light curve to be variable.
    """
    def __init__(self, shared_data):
        super().__init__(shared_data)
        self.Data = ['magnitude', 'error']

    def fit(self, data):
        magnitude = data[0]
        error = data[2]

        mean_mag = np.mean(magnitude)
        nepochs = float(len(magnitude))

        chi = np.sum((magnitude - mean_mag)**2. / error**2.)
        p_chi = chi2.cdf(chi, (nepochs-1))

        return p_chi


class ExcessVar(Base):
    """
    Calculate the excess variance,which is a measure of the intrinsic variability amplitude.
    """
    def __init__(self, shared_data):
        super().__init__(shared_data)
        self.Data = ['magnitude', 'error']

    def fit(self, data):
        magnitude = data[0]
        error = data[2]

        mean_mag = np.mean(magnitude)
        nepochs = float(len(magnitude))

        a = (magnitude-mean_mag)**2
        ex_var = (np.sum(a-error**2) / (nepochs * (mean_mag ** 2)))

        return ex_var
