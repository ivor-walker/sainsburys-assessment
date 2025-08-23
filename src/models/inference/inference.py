"""
Router to chosen inference model - instead of main needing to know which model to use, this class abstracts that away

Inherits from BaseModel, which provides the generic public model methods (training, predicting, etc) for this public-facing class

Functionally a duplicate of PredictionModel - would be overkill to create a base "Router" class from which both routers inherit
"""

from src.models.base_model import BaseModel

from src.models.inference.glmm import GLMm

class InferenceModel(BaseModel):
    def __init__(self,
        model_name: str = "GLMm",
    ):
        name_model_dict = {
            "GLMm": GLMm,
        }
        if model_name not in name_model_dict:
            raise ValueError(f"Model {model_name} is not supported.")
        
        super().__init__(name_model_dict[model_name]())
