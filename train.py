import os
from dotenv import load_dotenv
load_dotenv()

from src.data.data_prep import prepare_training_data

import eda.data_vis as data_vis

from src.models.base import Model

train_data, eval_data, scaling_info = prepare_training_data(
    product_details = os.getenv("PRODUCT_DETAILS_LOCATION"),
    catalogue_discontinuation = os.getenv("CATALOGUE_DISCONTINUATION_LOCATION"),
    load_training_data = os.getenv("TRY_LOADING_TRAINING_DATA", "False") == "True",
    train_data_loc = os.getenv("TRAINING_DATA_LOCATION"),
    eval_data_loc= os.getenv("EVALUATION_DATA_LOCATION"),
    scaling_info_loc = os.getenv("SCALING_INFO_LOCATION"),
    categorical_info_loc = os.getenv("CATEGORICAL_INFO_LOCATION"),
    save_training_data = os.getenv("SAVE_TRAINING_DATA", "False") == "True",
)

if os.getenv("SHOW_EDA", "False") == "True":
    print("Showing EDA visualisations.")
    data_vis.show_all_vis(train_data)

models = [
    Model(model_type = "linear"),
    # TODO implement other models
    # Model(model_type = "tree"),
    # Model(model_type = "nn"),
]

models_dir = os.getenv("MODELS_DIRECTORY", "models")

for model in models:
    model_path = f"{models_dir}/{model.get_model_name()}.model"

    model.train(train_data) 

    model.save(model_path)

    model.diagnose(train_data)

    model.eval(eval_data)
