"""
Perform inference from a pre-trained model
"""
import os
from dotenv import load_dotenv
load_dotenv()

from src.data.data_prep import prepare_data

from src.models.base import Model


data, scaling_info = prepare_data(
	product_details = os.getenv("PRODUCT_DETAILS_LOCATION"),
    catalogue_discontinuation = os.getenv("CATALOGUE_DISCONTINUATION_LOCATION"),
    load_processed_data = os.getenv("TRY_LOADING_PROCESSED_DATA", "False") == "True",
    processed_data_loc = os.getenv("PROCESSED_DATA_LOCATION"),
    scaling_info_loc = os.getenv("SCALING_INFO_LOCATION"),
    save_processed_data = os.getenv("SAVE_DATA_AFTER_PROCESSING", "False") == "True",
	categorical_info_loc = os.getenv("CATEGORICAL_INFO_LOCATION")
)

breakpoint()

inference_model = Model(model_type = "inference")
prediction_model = Model(model_type = "prediction")

for model in [inference_model, prediction_model]:
    model_path = f"{models_dir}/{model.get_model_name()}.model"

    model.load(model_path)
    
    model.predict(data)
