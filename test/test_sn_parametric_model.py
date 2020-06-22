import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from late_classifier.features.extractors.sn_model_tf import SNModel
from late_classifier.features.extractors.sn_parametric_model_computer import SNParametricModelExtractor

import time
import cProfile


snmodel = SNModel()
# times = np.linspace(-50, 200, 400)
# flux = snmodel(times)
# 
# flux2 = flux * 1.2
# times2 = times * 1.6

# print(snmodel.get_model_parameters())
# loss_value = snmodel.fit(times2, flux2)
# print(snmodel.get_model_parameters())
# print(loss_value)

# prediction = snmodel(times)
# plt.plot(times, prediction, 'r')
# plt.plot(times2, flux2, '--')
# plt.show()

detections = pd.read_pickle('../paper_paula/dataset_detections.pkl')
print(detections.head())
extractor = SNParametricModelExtractor()
time_list = []
for loop in range(2):
    for oid in ['ZTF18abdkkwa', 'ZTF18acrtvmm', 'ZTF18abokyfk']:
        lc0 = detections.loc[oid]
        lc0_r = lc0[lc0.fid == 1]

        t0 = time.time()
        #pr = cProfile.Profile()
        #pr.enable()
        features = extractor.compute_features(lc0_r)
        #pr.disable()
        #pr.dump_stats('sn_model.prof')
        #exit()
        tf = time.time() - t0
        time_list.append(tf)
print(np.array(time_list).mean(), 'seconds per sne')
