from src.data.data_prep import prepare_data

import eda.data_visualisation as data_viz

from src.models.base import Model

train_data, eval_data = prepare_data(
    product_details = "data/ProductDetails.csv",
    catalogue_discontinuation = "data/CatalogueDiscontinuation.csv"
)

# data_viz.visualise_data(data)

inference_model = Model(model_type = "inference")
prediction_model = Model(model_type = "prediction")

load_model = True
save_model = True

for model in [inference_model, prediction_model]:
    model_path = f"models/{model.get_model_name()}.model"

    try:
        if load_model == False:
            raise Exception("Model loading disabled, training new model.")

        model.load(model_path)
    
    except Exception as e:
        print(str(e))

        model.train(train_data) 

        if save_model:
            model.save()

    model.diagnose(train_data)

    model.eval(eval_data)
