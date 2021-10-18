import unittest
import numpy as np
from lc_classifier.features.extractors.sn_model_tf import SNModel
from lc_classifier.features.extractors.sn_parametric_model_computer import SPMExtractorPhaseII
from lc_classifier.features import ZTFLightcurvePreprocessor
from forced_photometry_data.txt_lc_parser import parse_lightcurve_txt, filter_name_to_int
import os


class TestSPM(unittest.TestCase):
    def setUp(self) -> None:
        self.sn_model = SNModel()

    def test_run(self):
        times = np.linspace(-50, 200, 400)
        flux = self.sn_model(times)
        flux2 = flux * 1.2
        times2 = times * 1.6
        loss_value = self.sn_model.fit(times2, flux2)
        prediction = self.sn_model(times)
        self.assertEqual(len(times), len(prediction))
        print(loss_value)


class TestSPMExtractorPhaseII(unittest.TestCase):
    def setUp(self) -> None:
        this_dir = os.path.dirname(os.path.abspath(__file__))
        example_csv_filename = os.path.join(
            this_dir,
            "../../forced_photometry_data/forcedphotometry_req00010458_lc.txt"
        )
        self.light_curve_df = parse_lightcurve_txt(example_csv_filename)
        self.light_curve_df['mjd'] = self.light_curve_df['jd'] - 2400000.5
        self.light_curve_df['difference_flux'] = self.light_curve_df['forcediffimflux']
        self.light_curve_df['difference_flux_error'] = self.light_curve_df['forcediffimfluxunc']
        self.light_curve_df['fid'] = [
            filter_name_to_int[name] for name in self.light_curve_df['filter']]
        self.light_curve_df['oid'] = 'ZTF21aaomuka'
        self.light_curve_df.set_index('oid', inplace=True)
        self.bands = [1, 2]
        self.preprocessor = ZTFLightcurvePreprocessor()
        self.light_curve_df = self.preprocessor.rename_columns_detections(
            self.light_curve_df)

    def test_fit(self):
        extractor = SPMExtractorPhaseII(self.bands)
        features = extractor.compute_features(self.light_curve_df)
        self.assertFalse(np.any(np.isnan(features.values)))
