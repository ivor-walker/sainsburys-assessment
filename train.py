import os
from dotenv import load_dotenv
load_dotenv()

from src.data.data_prep import prepare_training_data
import eda.data_vis as data_vis

from src.models.base import Model

train_data, eval_data, test_data, scaling_info = prepare_training_data(
    product_details = os.getenv("PRODUCT_DETAILS_LOCATION"),
    catalogue_discontinuation = os.getenv("CATALOGUE_DISCONTINUATION_LOCATION"),
    load_training_data = os.getenv("TRY_LOADING_TRAINING_DATA", "False") == "True",
    train_data_loc = os.getenv("TRAINING_DATA_LOCATION"),
    eval_data_loc= os.getenv("HYPERPARAMETER_EVALUATION_DATA_LOCATION"),
    test_data_loc = os.getenv("TEST_DATA_LOCATION"),
    scaling_info_loc = os.getenv("SCALING_INFO_LOCATION"),
    categorical_info_loc = os.getenv("CATEGORICAL_INFO_LOCATION"),
    save_training_data = os.getenv("SAVE_TRAINING_DATA", "False") == "True",
    train_on_sample = os.getenv("RESOURCE_CONSTRAINED_TRAINING_ON_SAMPLE", "False") == "True",
)

if os.getenv("SHOW_EDA", "False") == "True":
    print("Showing EDA visualisations.")
    data_vis.show_all_vis(train_data, scaling_info)

models = [
    Model(model_type = "tree"),
    # TODO implement other models as needed
    # Model(model_type = "linear"),
    # Model(model_type = "nn"),
]

models_dir = os.getenv("MODELS_DIRECTORY", "models")

for model in models:
    model.train(train_data, eval_data)

    model.save(models_dir)
    model.load(train_data, models_dir)

    model.diagnose()

    print(model.predict(test_data))
