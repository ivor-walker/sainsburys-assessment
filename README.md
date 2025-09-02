# Introduction
Solution to the Sainsbury's Nectar360 Data Science team's challenge.

## Instructions for running
### Requirements
By default, the solution looks for data in the data/ directory - please ensure the CatalogueDiscontinuation.csv and ProductDetails.csv files are present. Alternate locations and file names can be specified in the .env file.

### Running the code
To train the model, run train.py. To make predictions on new data, run predict.py - although this does not currently produce proper predictions, see the Conclusion section below.

# Methodology
## Modelling
Our data is clustered - CatalogueDiscontinuation.csv contains the same data (progression of WeeksOut from 24 to 1) for each product, so are repeatedly measuring the same variables. General purpose statistical and machine learning models assume that each row is independent, but in our case only the products are independent from each other and we have a time series inside each product. This restricts our model selection designed for this clustered data, predominantly the linear Generalised Estimating Equations (GEE), the linear-tree hybrid GPBoost, and a neural network. we are unable to consider the neural network because of computational and time constraints.

Another problem is the lack of data on when the decision to discontinue a product during its run is made. Different products have different lead times (i.e. how far in advance the product needs to be shipped to stores before it is available for sale), so products with longer lead times require an earlier decision to discontinue. Although we have a predictor of discontinuation ("status") for each week, we do not know when the decision to discontinue a product was made. We can infer this information using a product's metadata (e.g. a particular supplier may have long lead times from their particular supply chains, or non-domestic products may have longer lead times due to shipping), but these relationships are highly non-linear and likely have complex interactions. Although a GEE is appealing due to its inferential capabilities (i.e. we could make clear comments on the causes of differences in discontinuing dates), GEEs are unable to capture these dynamics so we need a more flexible model like GPBoost.

## Feature engineering
The discontinuation status does not change at all during a product's run, and for some products (see src.data.data\_prep: add\_product\_dynamics()) the status also does not change during the run. We can exploit this for a simple rules-based prediction (if status is always 0, predict 0; if status is always 1, predict 1). However, the scope of this rules-based prediction is limited as we only know if a product has a consistent status at weeks\_out = 1 which defeats the whole purpose of this exercise (i.e. predicting discontinuation in advance). 

We add features that capture changes and dynamics in status (e.g. how many times a product status has changed, when was the most recent change) that the model can use to understand which status is most predictive for each product. To facilitate this without data leakage, we implemented a rolling window which only uses data from weeks\_out > k to predict the discontinuation status at weeks\_out = k. Its implementation enables us to add other useful dynamics of other variables (e.g. sales dynamics that can be be used to guess remaining runtime).

Alongside these dynamics and more standard feature transformation processes (e.g. adding differentials or logs), we added a feature called "growth" that represents the expected growth of a product (i.e. future sales / past sales).

# Results
Unfortunately, the data exploration, modelling and feature engineering required to work on this data was extensive, and I lacked the time and computational resources to properly fine-tune the final model and the model did not fit at all. Solving this issue would be trivial with more computational resources as we already have an expansive grid search for hyperparameters set up (see src.models.gpboost: train()). Even training on a small fraction of the data proved too computationally expensive for my current resources.

# Conclusion
This project was an interesting challenge in dealing with clustered data and time series data, and the feature engineering required to extract useful information from the data. The final model did not fit due to computational constraints, but the methodology and feature engineering are sound and would quickly produce good results with more computational resources.

Once we have a fitted model, we could further improve the model by engineering more features relating to the specific product's performance in other catalogues. Currently, we only have a count of the number of catalogues a product has appeared in, but the rules-based prediction also fails for products that have appeared in multiple catalogues but have a consistent status. This indicates there is something about the product's performance in other catalogues that is predictive of its discontinuation status, but a more extensive EDA would be required to understand what this is.

Although the feature engineering and modelling frameworks I produced were robust and render adding new features or changing models trivial, I spent too much time on this and not enough time on the final model or on the EDA. I thought that GPBoost would be comparable in speed to LightGBM, and did not anticipate that it would be so computationally expensive to train because I have not used this more experimental framework before. Although it would not have been performant or inferentially valid, I should have tried a LightGBM model as a baseline to ensure I had a working model and noted the limitations of this approach in the report instead of trying to make GPBoost work.
