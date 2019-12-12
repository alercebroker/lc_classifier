import numpy as np
import pandas as pd


class DetectionsPreprocessor:
    def __init__(self):
        self.necessary_columns = ['mjd', 'ra', 'dec', 'magpsf_corr', 'sigmapsf_corr']
        self.max_sigma = 1.0
        self.rb_threshold = 0.55

    def preprocess_detections(self, detections):
        detections = self.discard_invalid_value_detections(detections)
        detections = self.drop_duplicates(detections)
        detections = self.discard_noisy_detections(detections)
        detections = self.discard_bogus(detections)
        return detections

    def discard_invalid_value_detections(self, detections):
        # detections = detections.replace(['Infinity'], np.inf)
        detections = detections.replace([np.inf, -np.inf], np.nan)
        valid_alerts = detections[self.necessary_columns].notna().all(axis=1)
        detections = detections[valid_alerts.values]
        detections[self.necessary_columns] = detections[self.necessary_columns].apply(pd.to_numeric)
        return detections

    def drop_duplicates(self, detections):
        detections = detections.copy()
        detections['oid'] = detections.index
        detections = detections.drop_duplicates(['oid', 'mjd'])
        detections = detections[[col for col in detections.columns if col != 'oid']]
        return detections

    def discard_noisy_detections(self, detections):
        detections = detections[
            (detections.sigmapsf_corr > 0.0)
            & (detections.sigmapsf_corr < self.max_sigma)]
        return detections

    def discard_bogus(self, detections):
        detections = detections[detections.rb >= self.rb_threshold]
        return detections
