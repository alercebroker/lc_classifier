import requests
import pandas as pd
from pandas.io.json import json_normalize


class AlerceAPI(object):
    def __init__(self, baseurl):

        self.baseurl = baseurl

    def query(self, params):

        # show api results
        r = requests.post(url = "%s/query" % self.baseurl, json = params)
        df = pd.DataFrame(r.json())
        query_results = json_normalize(df.result)
        query_results.set_index('oid', inplace=True)
        return query_results

    def get_sql(self, params):

        r = requests.post(url = "%s/get_sql" % self.baseurl, json = params)
        return r.content

    def get_detections(self, params):

        # show api results
        r = requests.post(url = "%s/get_detections" % self.baseurl, json = params)
        df = pd.DataFrame(r.json())
        detections = json_normalize(df.result.detections)
        detections.set_index('candid', inplace=True)
        return detections

    def get_non_detections(self, params):

        # show api results
        r = requests.post(url = "%s/get_non_detections" % self.baseurl, json = params)
        df = pd.DataFrame(r.json())
        detections = json_normalize(df.result.non_detections)
        detections.set_index('mjd', inplace=True)
        return detections

    def get_stats(self, params):

        # show api results
        r = requests.post(url = "%s/get_stats" % self.baseurl, json = params)
        df = pd.DataFrame(r.json())
        stats = json_normalize(df.result.stats)
        stats.set_index('oid', inplace=True)
        return stats

    def get_probabilities(self, params):

        # show api results
        r = requests.post(url = "%s/get_probabilities" % self.baseurl, json = params)
        early = json_normalize(r.json()["result"]["probabilities"]["early_classifier"])
        early.set_index("oid", inplace=True)
        late = json_normalize(r.json()["result"]["probabilities"]["random_forest"])
        late.set_index("oid", inplace=True)
        return early, late

    def get_features(self, params):

        # show api results
        r = requests.post(url = "%s/get_features" % self.baseurl, json = params)
        features = json_normalize(r.json())
        features.set_index('oid', inplace=True)
        return features

    def plotstamp(self, oid, candid):
        science = "http://avro.alerce.online/get_stamp?oid=%s&candid=%s&type=science&format=png" % (oid, candid)
        images="""
        &emsp;&emsp;&emsp;&emsp;&emsp;
        Science
        &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;
        Template
        &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;
        Difference
        <div class="container">
        <div style="float:left;width:20%%"><img src="%s"></div>
        <div style="float:left;width:20%%"><img src="%s"></div>
        <div style="float:left;width:20%%"><img src="%s"></div>
        </div>
        """ % (science, science.replace("science", "template"), science.replace("science", "difference"))
        display(HTML(images))


api = AlerceAPI("http://ztf.alerce.online")
oid = 'ZTF18aaiopei'
detections = api.get_detections({'oid': oid})
detections.reset_index(inplace=True)
detections = detections[['fid', 'mjd', 'oid', 'magpsf_corr', 'sigmapsf_corr']].copy()
detections.dropna(inplace=True)
detections.drop_duplicates(['oid', 'fid', 'mjd'], inplace=True)
detections.set_index('oid', inplace=True)
detections.sort_values('mjd', inplace=True)
detections = detections.iloc[:239].copy()

detections.to_pickle(f'{oid}_detections.pkl')
