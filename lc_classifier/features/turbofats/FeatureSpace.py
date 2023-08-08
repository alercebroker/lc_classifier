import numpy as np
import pandas as pd
from . import FeatureFunctionLib
from numba import jit
import warnings


@jit(nopython=True)
def is_sorted(a):
    for i in range(a.size-1):
        if a[i+1] < a[i]:
            return False
    return True


class FeatureSpace(object):
    def __init__(self, feature_list, data_column_names=None, extra_arguments: dict = {}):
        self.feature_objects = []
        self.feature_names = []
        self.shared_data = {}
        if data_column_names is None:
            self.data_column_names = ['magnitude', 'time', 'error']
        else:
            self.data_column_names = data_column_names

        for feature_name in feature_list:
            feature_class = getattr(FeatureFunctionLib, feature_name)
            if feature_name in extra_arguments:
                feature_instance = feature_class(
                    self.shared_data,
                    **extra_arguments[feature_name]
                )
            else:
                feature_instance = feature_class(self.shared_data)
            self.feature_objects.append(feature_instance)
            if feature_instance.is1d():
                self.feature_names.append(feature_name)
            else:
                self.feature_names += feature_instance.get_feature_names()

    def __lightcurve_to_array(self, lightcurve):
        return lightcurve[self.data_column_names].values.T

    def calculate_features(self, lightcurve):
        n_objects = len(lightcurve.index.unique())
        if n_objects > 1:
            raise Exception('TurboFATS cannot handle more than one lightcurve simultaneously')
        elif n_objects == 0 or len(lightcurve) <= 5:
            df = pd.DataFrame(columns=self.feature_names)
            df.index.name = 'oid'
            return df

        oid = lightcurve.index.values[0]
        if not is_sorted(lightcurve[self.data_column_names[1]].values):
            lightcurve = lightcurve.sort_values(self.data_column_names[1])
            
        lightcurve_array = self.__lightcurve_to_array(lightcurve)
                    
        results = []
        self.shared_data.clear()
        for feature in self.feature_objects:
            try:
                result = feature.fit(lightcurve_array)
            except Exception as e:
                warnings.warn('Exception when computing turbo-fats feature: '+str(e))
                if feature.is1d():
                    result = np.NaN
                else:
                    result = [np.NaN] * len(feature.get_feature_names())
            if feature.is1d():
                results.append(result)
            else:
                results += result
        results = np.array(results).reshape(1, -1).astype(float)
        df = pd.DataFrame(results, columns=self.feature_names, index=[oid])
        df.index.name = 'oid'
        return df
