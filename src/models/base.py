"""
Model - sets encapsulated model, and provides generic public methods to interact with the encapsulated model
"""
from src.models.glmm import GLMm
from src.models.lightgbm import LightGBM

class Model:
    """
    Constructor: attempt to fetch and set encapsulated model
    """
    def __init__(self,
        model_type: str, 
        model_name: str = None,
    ):
        if model_type == "inference":
            name_model_dict = {
                "GLMm": GLMm,
            }
        elif model_type == "prediction":
            name_model_dict = {
                "LightGBM": LightGBM,
            }
        else:
            raise ValueError(f"Model type {model_type} is not supported.")
        
        self.__set_model_name(model_type, model_name)

        try: 
            self.__model = name_model_dict[self.__model_name]()
        except KeyError:
            raise ValueError(f"Model name {self.__model_name} is not supported for model type {model_type}.")
    
    def __set_model_name(self,
        model_type: str,
        model_name: str,
        default_inference: str = "GLMm",
        default_prediction: str = "LightGBM",
    ):
        if model_name is None:
            if model_type == "inference":
                model_name = default_inference
            elif model_type == "prediction":
                model_name = default_prediction

        self.__model_name = model_name
    
    """
    Public methods to interact with the encapsulated model
    """
    def get_model_name(self):
        return self.__model_name

    def load(self, path: str):
        self.__model.load(path)
    
    def train(self, data):
        self.__model.train(data)

    def save(self, path: str):
        self.__model.save(path)

    def diagnose(self, data):
        return self.__model.diagnose(data)

    def predict(self, data):
        return self.__model.predict(data)

    def eval(self, data):
        # TODO
        pass
