from .hierarchical_extractor import *
from .sn_non_detections_refactor import *
from .turbofats_extractor import *
from .color_feature_extractor import *
from .sn_detections_extractor import *
from .galactic_coordinates_extractor import *
from .real_bogus_extractor import *
from .sg_score_extractor import *
from .mhps_extractor import *
from .iqr_extractor import *

features_mapping = {
    "iqr": IQRExtractor(),
    "mhps": MHPSExtractor(),
    "sgscore": SGScoreComputer(),
    "realbogus": RealBogusComputer(),
    "galactic": GalacticCoordinatesComputer(),
    "color": ColorFeatureExtractor(),
    "turbofats": TurboFatsFeatureExtractor(),
    "sn-det": SupernovaeDetectionFeatureComputer(),
    "sn-nondet": SupernovaeNonDetectionFeatureComputer(),
    #"hierarchical": HierarchicalFeaturesComputer()
}

__all__ = ['HierarchicalFeaturesComputer',
           'SupernovaeNonDetectionFeatureComputer',
           'SupernovaeDetectionFeatureComputer',
           'TurboFatsFeatureExtractor',
           'ColorFeatureExtractor',
           'GalacticCoordinatesComputer',
           'RealBogusComputer',
           'SGScoreComputer',
           'MHPSExtractor',
           'IQRExtractor']
