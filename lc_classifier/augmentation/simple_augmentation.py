import numpy as np
import pandas as pd


class Augmenter:
    def augment(
            self,
            detections: pd.DataFrame,
            non_detections: pd.DataFrame,
            labels: pd.DataFrame,
            multiplier: int):
        raise NotImplementedError('Augmenter is an abstract class')

    def _get_augmentable_oids(
            self,
            labels: pd.DataFrame,
            augmentable_classes):
        useful_rows = labels[
            labels.classALeRCE.isin(augmentable_classes)]
        return useful_rows.index.values


class ShortTransientAugmenter(Augmenter):
    def __init__(self):
        self.augmentable_classes = [
            'SNIa',
            'SNIbc',
            'SNII',
            'SNIIn',
            'SLSN'
        ]

    def augment(
            self,
            detections: pd.DataFrame,
            non_detections: pd.DataFrame,
            labels: pd.DataFrame,
            multiplier: int):
        augmentable_oids = self._get_augmentable_oids(labels, self.augmentable_classes)
        new_detections = []
        new_non_detections = []
        new_labels = []
        for template_oid in augmentable_oids:
            template_det_lc = detections.loc[[template_oid]].copy()
            if template_oid in non_detections.index.unique():
                template_non_det_lc = non_detections.loc[[template_oid]]
            else:
                template_non_det_lc = non_detections.loc[[]]
            template_det_lc.sort_values(by=['mjd'], inplace=True)

            possible_lengths = range(11, len(template_det_lc))
            new_lengths = np.random.choice(
                possible_lengths,
                min(len(possible_lengths), multiplier), replace=False)
            for i, new_length in enumerate(new_lengths):
                short_detections = template_det_lc.iloc[:new_length].copy()
                max_mjd = max(short_detections.mjd.values)
                short_non_detections = template_non_det_lc[template_non_det_lc.mjd < max_mjd].copy()

                new_oid = template_oid + f'_short_transient_augmenter_{i}'
                short_detections.index = [new_oid]*len(short_detections)
                short_non_detections.index = [new_oid]*len(short_non_detections)

                new_detections.append(short_detections)
                new_non_detections.append(short_non_detections)
                short_label = labels.loc[[template_oid]].copy()
                short_label.index = [new_oid]
                new_labels.append(short_label)

        new_detections = pd.concat(new_detections, axis=0)
        new_non_detections = pd.concat(new_non_detections, axis=0)
        new_labels = pd.concat(new_labels, axis=0)

        return new_detections, new_non_detections, new_labels
