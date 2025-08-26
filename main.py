import os
from dotenv import load_dotenv
load_dotenv()

from src.data.data_prep import prepare_data

import eda.data_vis as data_vis

from src.models.base import Model

train_data, eval_data = prepare_data(
    product_details = os.getenv("PRODUCT_DETAILS_LOCATION"),
    catalogue_discontinuation = os.getenv("CATALOGUE_DISCONTINUATION_LOCATION"),
)

if os.getenv("SHOW_EDA", "False") == "True":
    print("Showing EDA visualisations.")
    data_vis.show_all_vis(train_data)

inference_model = Model(model_type = "inference")
prediction_model = Model(model_type = "prediction")

load_model = os.getenv("TRY_LOADING_MODEL", "False") == "True"
save_model = os.getenv("SAVE_MODEL_AFTER_TRAINING", "False") == "True"
models_dir = os.getenv("MODELS_DIRECTORY", "models")

for model in [inference_model, prediction_model]:
    model_path = f"{models_dir}/{model.get_model_name()}.model"

    try:
        if load_model == False:
            raise Exception("Model loading disabled in environment variables.")

        model.load(model_path)
    
    except Exception as e:
        print(str(e))
        print("Training new model.")

        model.train(train_data) 

        if save_model:
            model.save(model_path)

    model.diagnose(train_data)

    model.eval(eval_data)
