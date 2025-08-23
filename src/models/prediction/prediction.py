"""
Router to chosen prediction model. See src/models/inference/inference.py for documentation.
"""

from src.models.base_model import BaseModel

from src.models.prediction.lightgbm import LightGBM

class PredictionModel(BaseModel):
    def __init__(self,
        model_name: str = "LightGBM",
    ):
        name_model_dict = {
            "LightGBM": LightGBM,
        }
        if model_name not in name_model_dict:
            raise ValueError(f"Model {model_name} is not supported.")
        
        super().__init__(name_model_dict[model_name]())
