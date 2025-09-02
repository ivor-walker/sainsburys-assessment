"""
Model - sets encapsulated model, and provides generic public methods to interact with the encapsulated model
"""
from src.models.gpboost import GPBoost
from src.models.gee import GEE
from src.models.mlp import MLP

class Model:

    """
    Constructor: attempt to fetch and set encapsulated model
    """
    def __init__(self,
        model_type: str, 
        model_name: str = None,
    ):
        
        self.__set_model_name(model_type, model_name)
        
        name_model_dict = {
            "GEE": GEE,
            "GPBoost": GPBoost,
            "GatedMultiHeadMLP": MLP,
        }

        try: 
            self.__model = name_model_dict[self.__model_name]()
        except KeyError:
            raise ValueError(f"Model name {self.__model_name} is not supported for model type {model_type}.")
    
    def __set_model_name(self,
        model_type: str,
        model_name: str,
    ):
        if model_name is not None:
            self.__model_name = model_name     

        elif model_type == "linear":
            self.__model_name = "GEE"

        elif model_type == "tree":
            self.__model_name = "GPBoost"

        elif model_type == "nn":
            self.__model_name = "GatedMultiHeadMLP"

        else:
            raise ValueError(f"Model type {model_type} is not supported.")

    """
    Public methods to interact with the encapsulated model
    """
    def get_model_name(self):
        return self.__model_name

    def load(self, data, path: str):
        self.__model.load(data, path)
    
    def train(self, train_data, eval_data):
        self.__model.train(train_data, eval_data)

    def save(self, path: str):
        self.__model.save(path)

    def diagnose(self):
        return self.__model.diagnose()

    def predict(self, data):
        return self.__model.predict(data)

    def eval(self, data):
        # TODO
        pass
