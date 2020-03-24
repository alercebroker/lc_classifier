import os
import pandas as pd

from tqdm import tqdm

MODEL_PATH = os.path.dirname(os.path.abspath(__file__))
PICKLE_PATH = os.path.join(MODEL_PATH, "pickles")


class HierarchicalRF:
    def __init__(self):
        self.pickles = ["periodic.pkl", "root.pkl", "stochastic.pkl", "transient.pkl"]
        self.url_model = "https://assets.alerce.online/pipeline/hierarchical_rf_0.2/"
        self.model = {
            "root": {},
            "children": {}
        }
        self.init_model()

    def download_model(self):
        if not os.path.exists(PICKLE_PATH):
            os.mkdir(PICKLE_PATH)
        for pkl in tqdm(self.pickles):
            tmp_path = os.path.join(PICKLE_PATH, pkl)
            if not os.path.exists(tmp_path):
                command = f"wget {self.url_model}{pkl} -O {tmp_path}"
                os.system(command)

    def init_model(self):
        self.download_model()
        root_path = os.path.join(PICKLE_PATH, "root.pkl")
        periodic_path = os.path.join(PICKLE_PATH, "periodic.pkl")
        stochastic_path = os.path.join(PICKLE_PATH, "stochastic.pkl")
        transient_path = os.path.join(PICKLE_PATH, "transient.pkl")

        root_dict = pd.read_pickle(root_path)
        periodic_dict = pd.read_pickle(periodic_path)
        stochastic_dict = pd.read_pickle(stochastic_path)
        transient_dict = pd.read_pickle(transient_path)

        self.model["root"]["model"] = root_dict["rf_model"]
        self.model["root"]["classes"] = root_dict['order_classes']
        self.model["features"] = root_dict["features"]

        self.model["children"] = {
            "Stochastic": {
                "model": stochastic_dict["rf_model"],
                "classes": stochastic_dict["order_classes"]
            },
            "Periodic": {
                "model": periodic_dict["rf_model"],
                "classes": periodic_dict["order_classes"]
            },
            "Transient": {
                "model": transient_dict["rf_model"],
                "classes": transient_dict["order_classes"]
            }
        }

    def predict(self, input_features, pipeline=False):
        prob_root = pd.DataFrame(self.model["root"]["model"].predict_proba(input_features),
                                 columns=self.model["root"]["classes"])

        prob_children = []
        resp_children = {}
        for key in self.model["children"]:
            prob_child = pd.DataFrame(self.model["children"][key]["model"].predict_proba(input_features),
                                      columns=self.model["children"][key]["classes"])
            if pipeline:
                resp_children[key] = prob_child.iloc[0].to_dict()
            prob_child = prob_child.mul(prob_root[key].values, axis="rows")
            prob_children.append(prob_child)
        prob_all = pd.concat(prob_children, axis=1, sort=False)

        if pipeline:

            return {
                "hierarchical": {
                    "root": prob_root.iloc[0].to_dict(),
                    "children": resp_children
                },
                "probabilities": prob_all.iloc[0].to_dict(),
                "class": prob_all.iloc[0].idxmax()
            }

        return prob_children





