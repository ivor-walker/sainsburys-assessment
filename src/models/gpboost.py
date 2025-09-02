"""
Variation of gradient boosting for structured train_data
Us
"""
from src.models.interface import InterfaceModel

from src.data.data_prep import combine_grouping_columns
from src.data.data_prep import get_sample_products

import gpboost as gpb

import numpy as np

class GPBoost(InterfaceModel):
    def __init__(self):
        self.__booster_parameter_grid = { 
            'learning_rate': [0.001, 0.01, 0.1, 1, 10], 
            'min_data_in_leaf': [1, 10, 100, 1000],
            'max_depth': [-1], # -1 means no depth limit as we tune 'num_leaves'. Can also additionally tune 'max_depth', e.g., 'max_depth': [-1, 1, 2, 3, 5, 10]
            'num_leaves': 2**np.arange(1,10),
            'lambda_l2': [0, 1, 10, 100],
            'line_search_step_length': [True, False],
        }
        
        self.__booster_default_parameters = {
            "num_threads": 8,
            "verbose": 1,
        }
        
        # Naive guess of parameters, overriden by grid search
        self.__booster_parameters = {
            "learning_rate": 0.1,
            "min_data_in_leaf": 50,
            "num_leaves": 31,
            "lambda_l2": 0,
            "line_search_step_length": False,
        }

        self.__likelihood = "bernoulli_logit"

    def train(self, train_data, eval_data,
        group_col: str = "product_catalogue_group",
        grid_search: bool = False,

    ):
        print("Preparing train_data for GPBoost - this is highly memory intensive...")
        train_X, train_y = train_data
        del train_data

        eval_X, eval_y = eval_data
        del eval_data

        train_X = combine_grouping_columns(train_X)
        group_col = [col for col in train_X.columns if col == group_col][0]
        group_data = train_X[group_col]
        train_X = train_X.drop(group_col)
        
        train_X, train_y, group_data = train_X.to_pandas(), train_y.to_series().to_list(), group_data.to_list()
        n = len(train_y)
        
        self.__random_effect_model = gpb.GPModel(group_data = group_data, likelihood = self.__likelihood)
        del group_data
         
        # print("Fitting GPModel model for random effects - this may take a while...")
        # self.__random_effect_model.fit(X = train_X, y = train_y)
        
        gp_train_data = gpb.Dataset(train_X, train_y)
        del train_X, train_y
        
        eval_X, eval_y = eval_X.to_pandas(), eval_y.to_series().to_list()
        gp_eval_data = gpb.Dataset(eval_X, eval_y)
        del eval_X, eval_y

        if grid_search:
            print("Tuning GPBoost hyperparameters for fixed effects - this may take a while...")
                    
            self.__booster_parameter_grid['max_bin'] = [
                250, 
                500, 
                1000, 
                np.min([10000, n])
            ]
            
            self.__booster_param_grid = gpb.grid_search_tune_parameters(
                param_grid = self.__booster_parameter_grid,
                params = self.__booster_default_parameters,
                train_set = gp_train_data,
                nfold = 5,
                gp_model = self.__random_effect_model,
                metric = "binary_logloss",
            )
        
        print("Training GPBoost model for fixed effects - this may take a while...")
        self.__eval_results = {}
        self.__fixed_effect_booster = gpb.train(
            self.__booster_parameters,
            train_set = gp_train_data,
            gp_model = self.__random_effect_model,
            # valid_sets = gp_eval_data,
            # early_stopping_rounds = 50,
            # evals_result = self.__eval_results,
        )

        print("GPModel and GPBoost for random and fixed effects trained.")

    def predict(self, test_X):
        test_X = combine_grouping_columns(test_X)
        group_col = [col for col in test_X.columns if "group" in col][0]
        group_data = test_X[group_col]
        #test_X = test_X.drop(group_col) 
        
        test_X = test_X.to_pandas()
        to_categorical = self.__fixed_effect_booster.pandas_categorical
        if len(to_categorical) > 0:
            for target_col in to_categorical:
                if any([target_col in c for c in test_X.columns]):
                test_X[col] = test_X[col].astype('category')

        breakpoint()

        preds = self.__fixed_effect_booster.predict(
            test_X, 
        )

    def diagnose(self):
        importance = self.__fixed_effect_booster.feature_importance(importance_type='gain')
        feature_names = self.__fixed_effect_booster.feature_name()
        feature_importance = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
        print(feature_importance)

    def load(self, data,
        path: str
    ):
        print(f"Trying to load GPBoost model from {path}...")
        # self.__random_effect_model = gpb.GPModel(model_file = f"{path}/model.json")
        data_X, data_y = data[0].to_pandas(), data[1].to_series().to_list()
        data = gpb.Dataset(data_X, data_y)
        self.__fixed_effect_booster = gpb.Booster(model_file = f"{path}/booster.json", train_set = data)
        print("GPBoost model loaded.")

    def save(self,
        path: str
    ):
        print(f"Saving GPBoost model to {path}...")
        self.__random_effect_model.save_model(filename = f"{path}/model.json")
        self.__fixed_effect_booster.save_model(filename = f"{path}/booster.json")
        print("GPBoost model saved.")
