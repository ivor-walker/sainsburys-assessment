from src.data.data_prep import prepare_data

import eda.data_visualisation as data_viz

from src.models.inference.base import InferenceModel
from src.models.prediction.base import PredictionModel

train_data, eval_data = prepare_data(
    product_details = "data/ProductDetails.csv",
    catalogue_discontinuation = "data/CatalogueDiscontinuation.csv"
)

# data_viz.visualise_data(data)

inference_model = InferenceModel()
prediction_model = PredictionModel();

load_model = True
for model in [inference_model, prediction_model]:
    if load_model:
        model.load(
            model_path = "models/model.pkl",
            data = train_data
        )
    else:
        model.train(train_data) 
        model.save()

    model.diagnose()

    model.eval(eval_data)
