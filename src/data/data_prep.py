"""
Data preprocessing functions
"""

import polars as pl

"""
Final pipeline to be imported by main
"""
def prepare_data(
    product_details: str,
    catalogue_discontinuation: str,
    load_processed_data: bool,
    train_data_loc: str,
    eval_data_loc: str,
    save_processed_data: bool,
    scaling_info_loc: str,
    target_col: str = "discontinued",
    group_cols: list = ["product", "catalogue"],
) -> tuple:

    try:
        if not load_processed_data:
            raise Exception("Loading processed data disabled in environment variables")
        
        print("Loading processed data...")
        train_data = pl.read_parquet(train_data_loc)
        eval_data = pl.read_parquet(eval_data_loc)
        scaling_info = pl.read_parquet(scaling_info_loc)

    except Exception as e:
        print(str(e))
        print("Reading and processing data...")

        data = __read_data(product_details, catalogue_discontinuation) 

        data = __clean_data(data)
        
        # Add revenue as an interaction instead of in-data specifically, to avoid perfect collinearity with price and sales
        # data = __add_revenue(data)
        
        data = __add_estimated_growth(data)

        data = __add_demand_dynamics(data)

        data = __add_previous_run_info(data)
        
        print("Adding product dynamics - this may take a while...")
        data = __add_product_dynamics(data)
        
        print("Transforming for training...")
        data = __transform_for_training(data, target_col = target_col, group_cols = group_cols)

        train_data, eval_data = __train_eval_split(data)
        
        # Get scaling info from train_dataing data, and apply to both train_data and eval_data
        scaling_info = __get_scaling_info(train_data)
        
        train_data = __apply_scaling(train_data, scaling_info)
        eval_data = __apply_scaling(eval_data, scaling_info)
        
        train_data = __address_high_cardinality(train_data)
        
        print("Data processing complete")

        if save_processed_data:
            train_data.write_parquet(train_data_loc)
            eval_data.write_parquet(eval_data_loc)
            scaling_info.write_parquet(scaling_info_loc)
            print(f"Processed data saved to {train_data_loc}, {eval_data_loc}, {scaling_info_loc}")

    finally:
        # Update target and group col names after transform
        index_cols = ["product_group", "catalogue_group", "weeks_out"]

        __test_data(train_data, data_name = "train_data", group_cols = index_cols)
        __test_data(eval_data, data_name = "eval_data", group_cols = index_cols)
        
        __test_data(scaling_info, data_name = "scaling_info", group_cols = ["feature"])

        target_col = f"{target_col}_target"
        train_data = __seperate_dependent_variable(train_data, target = target_col) 
        eval_data = __seperate_dependent_variable(eval_data, target = target_col) 
        return train_data, eval_data, scaling_info

"""
Read and combine ProductDetails and CatalogueDiscontinuation data
"""
def __read_data(
    product_details: str,
    catalogue_discontinuation: str
) -> tuple:
    product_details_frame = pl.read_csv(product_details) 
    catalogue_discontinuation_frame = pl.read_csv(catalogue_discontinuation)

    return catalogue_discontinuation_frame.join(
        product_details_frame,
        left_on = "ProductKey",
        right_on = "ProductKey",
        how = "inner",
        validate = "m:1",
        maintain_order = "left",
        coalesce = True
    )

"""
Clean data by recasting columns, renaming, etc
"""
def __clean_data(
    data: pl.DataFrame
) -> pl.DataFrame:
    data = data.select(
        # Recast ProductKey, CatEdition, Supplier, Hierarchy to categorical
        pl.col("ProductKey").cast(pl.Utf8).cast(pl.Categorical(ordering = "lexical")).alias("product"),
        pl.col("CatEdition").cast(pl.Utf8).cast(pl.Categorical).alias("catalogue"),
        pl.col("Supplier").cast(pl.Utf8).cast(pl.Categorical).alias("supplier"),

        pl.col("HierarchyLevel1").cast(pl.Utf8).cast(pl.Categorical).alias("specific_category"),
        pl.col("HierarchyLevel2").cast(pl.Utf8).cast(pl.Categorical).alias("broad_category"),

        # Set DIorDOM to boolean
        pl.when(pl.col("DIorDOM") == "DOM").then(True).otherwise(False).alias("domestic"),

        # Leave seasonal unchanged
        pl.col("Seasonal").alias("seasonal"),

        # Recast SalesPerIncVAT to float32 (not all prices end in 99)
        pl.col("SalePriceIncVAT").cast(pl.Float32).alias("price"),

        # Take absolute values of WeeksOut (to avoid negative log behaviour) and recast to UInt8
        pl.col("WeeksOut").abs().cast(pl.UInt8).alias("weeks_out"),
        
        # Recast status to boolean
        pl.when(pl.col("Status") == "RI").then(False).otherwise(True).alias("range_out_weekly"),

        # Recast ForecastPerWeek, ActualsPerWeek as f32 to save memory
        pl.col("ForecastPerWeek").cast(pl.Float32).alias("forecasted_remaining_sales_weekly"),
        pl.col("ActualsPerWeek").cast(pl.Float32).alias("actual_completed_sales_weekly"),
        
        # Leave target variable DiscontinuedTF unchanged
        pl.col("DiscontinuedTF").alias("discontinued")
    )

    # Return sorted by product, catalogue, weeks_out reversed (latest week first)
    return data.sort(["product", "catalogue", "weeks_out"], descending = [False, False, True])

"""
Add forecasted and actual revenue
Typically these kinds of business decisions are profit-oriented, not sales-oriented. Since we don't have COGS for each product, we have to use revenue as a proxy for profit.
"""
def __add_revenue(
    data: pl.DataFrame
) -> pl.DataFrame:
    return data.with_columns(
        # Revenue = price * quantity
        (pl.col("price") * pl.col("forecasted_remaining_sales_weekly")).alias("forecasted_remaining_revenue_weekly"),
        (pl.col("price") * pl.col("actual_completed_sales_weekly")).alias("actual_completed_revenue_weekly")
    )

"""
Add estimates of growth
For a given decision point t, we have two datapoints: actual_completed, 0 to t - 1, and forecasted_remaining, t to n. Completed represents a single point estimate of the past, remaining represents a single point estimate of the future. 
A naive estimate of growth is the ratio between future and past, or forecasted_remaining / actual_completed. Growth above 1 means the product is expected to grow, growth below 1 means the product is expected to shrink.
"""
def __add_estimated_growth(
    data: pl.DataFrame
) -> pl.DataFrame:
    return data.with_columns(
        # Estimate of growth = future / past 
        (pl.col("forecasted_remaining_sales_weekly") / pl.col("actual_completed_sales_weekly")).alias("estimated_growth_sales_weekly"),
    )

"""
Add derivatives of demand and growth
We don't need to do this for revenue since price is constant during a catalogue run, so revenue dynamics are the same as sales dynamics
"""
def __add_demand_dynamics(
    data: pl.DataFrame
) -> pl.DataFrame:
    return data.with_columns(
        pl.col("actual_completed_sales_weekly").diff().alias("change_actual_completed_sales_weekly"),
        pl.col("forecasted_remaining_sales_weekly").diff().alias("change_forecasted_remaining_sales_weekly"),
        pl.col("estimated_growth_sales_weekly").diff().alias("change_estimated_growth_sales_weekly"),
    )

"""
Add constant info to the weekly data on how the product's range status has changed over time

range_out (status) represents most of our weekly info on whether a product is discontinued, and the performance of range_out over a product's run is predictive of discontinuation

For example, 3/4 of products are 'easy' cases where range_out has been consistent throughout the product's run up until weeks_out - products with completed runs at weeks_out = 1 are ~99% likely to match the discontinuation status

However, the scope of this rules-based prediction is limited because a) it does not work outside of weeks_out = 1, which does not work for our application where we need to make predictions at any weeks_out, and b) it does not work on products with a non-continuous run (e.g. range_out flips from False to True and back to False)

To solve both a) and b), we need to understand which weeks_out holds the predictive range_out. The time at which a product's discontinuation is decided is unknown and varies between products (e.g. a product with a higher lead time needs an earlier decision), so we need to add features that capture changes and dynamics in range_out that the model can use to understand which weeks_out is most predictive for each product

To construct these features and avoid data leakage, we need to create some rolling window mechanism whereby we can look at the product's range_out history up to the current weeks_out, but not beyond it. 

This ability allows us to create features that capture other variable's dynamics throughout a products lifetime, e.g. weeks_out or actual_completed_sales_weekly, which give context to the range_out dynamics, e.g. whether a product has flipped a lot relative to its run length. They also let us construct measures on forecast volatility that are better than the limited categorical explainers we have (i.e. seasonal, domestic)
"""
def __add_product_dynamics(
    data: pl.DataFrame,
    easy_predictions: bool = False,
    easy_prediction_accuracy: bool = False,
    catalogue_length: int = 24,
    final_week: int = 1,
    min_catalogue_length: int = 20,
) -> pl.DataFrame:
    # Create rolling window - O(n * m * k) where m = start_runtime - weeks_out and k = number of columns to cumulate, extremely expensive!!
    cum_cols = ["range_out_weekly", "weeks_out", "actual_completed_sales_weekly", "forecasted_remaining_sales_weekly"]
    cum_agg_data = get_product_cum_aggregate(data, cum_cols = cum_cols)
    
    # Add product status info
    product_run_info = __add_product_status_info(cum_agg_data)
    
    # Add weeks_out info
    product_run_info = __add_weeks_out_info(product_run_info)
    
    # Add product status features, dependent on weeks_out
    product_run_info = __add_weekly_product_status(product_run_info)

    # Drop list of flips once they become unecessary
    product_run_info = product_run_info.drop("product_out_flips")
    
    # Add volatility measures
    product_run_info = __add_volatility_measures(product_run_info)

    # Reset forecasts from list to single latest value
    product_run_info = product_run_info.with_columns(
        pl.col("forecasted_remaining_sales_weekly").list[-1].alias("forecasted_remaining_sales_weekly"),
    )
    
    # Add actual/forecasted sales info that can be used to infer how long the product will run for
    product_run_info = __add_sales_info(product_run_info)

    # Add rules-based predictions for easy cases 
    if easy_predictions:
        product_run_info = __add_easy_predictions(product_run_info)

    # Check accuracy of easy predictions
    if easy_predictions and easy_prediction_accuracy:
        product_run_info = __check_accuracy_easy_predictions(product_run_info) 

    # Join this extra product info onto original weekly data
    on_cols = ["product", "catalogue", "weeks_out"]
    drop_cols = [col for col in product_run_info.columns if col in data.columns and col not in on_cols]
    return data.join(
        product_run_info.drop(drop_cols),
        on = on_cols,
        how = "left",
        validate = "1:1",
    )

"""
Check accuracy of easy predictions
"""
def __check_accuracy_easy_predictions(
    product_run_info: pl.DataFrame
) -> pl.DataFrame:
    product_run_info = product_run_info.with_columns(
        (
            # Equals predicted value
            (pl.col("easy_pred_discontinue") == pl.col("discontinued")) |
            
            # Or no prediction made
            (pl.col("easy_pred_discontinue").is_null())

        ).alias("accuracy")
    )

    return product_run_info

"""
Add rules-based predictions for easy cases
"""
def __add_easy_predictions(
    data: pl.DataFrame,
) -> pl.DataFrame:
    force_false = (
        # Product has been around for a short time - always retain
        (pl.col("total_runtime").le(min_catalogue_length))
    )

    force_true = (
        # Product does not run to end of catalogue - always discontinue
        # Cannot be known in advance though
        # (pl.col("end_runtime") != final_week) |

        # Non-continuous run - always discontinue
        (pl.col("n_missing_runtimes") != 0)
    )

    force_unknown = (
        # Product appears in a previous catalogue - goto ML
        (pl.col("catalogue_count") > 0) |
        
        # If product isn't consistently one status - goto ML
        (pl.col("n_product_out_flips") != 0)
    )

    product_run_info = product_run_info.with_columns(
        pl.when(
            force_unknown
        ).then(
            pl.lit(None)
        ).when(
            force_true | pl.col("first_product_range_out")
        ).then(
            pl.lit(True)
        ).when(
            force_false | ~pl.col("first_product_range_out")
        ).then(
            pl.lit(False)
        ).alias("easy_pred_discontinue")
    )

    return product_run_info

"""
Add sales info that can be used to infer how long the product will run for
"""
def __add_sales_info(
    product_run_info: pl.DataFrame
) -> pl.DataFrame:
    product_run_info = product_run_info.with_columns(
        # Cumulative actual sales so far
        (pl.col("actual_completed_sales_weekly").list.sum()).alias("cumsum_actual_completed_sales"),
    )

    product_run_info = product_run_info.with_columns(
        # Average of cumulative actual sales so far
        (pl.col("cumsum_actual_completed_sales") / pl.col("weeks_in")).alias("average_cumsum_actual_completed_sales"),
    )

    product_run_info = product_run_info.with_columns(
        # Actual sales so far plus forecasted sales
        (pl.col("cumsum_actual_completed_sales") + pl.col("forecasted_remaining_sales_weekly")).alias("expected_total_sales"),
    )

    product_run_info = product_run_info.with_columns(
        # Ratio between forecast and cumsum average: 
        # As weeks_in approaches total_runtime, the cumsum dwarves the forecasted remaining and this approaches zero
        (pl.col("forecasted_remaining_sales_weekly") / pl.col("average_cumsum_actual_completed_sales")).alias("ratio_forecast_cumsum_average"),
    
        # Ratio between actual sales so far and expected total sales: how much of the expected sales have we already seen?
        # More complicated measure of how far through the product's run we are, but in sales terms rather than time terms
        # As weeks_in approaches total_runtime, the cumsum dwarves the forecasted remaining and this goes from 0 (low cumulative sales : high expected sales because 1 week forecast is relatively high compared to 1 weeks cumulative sales) to 1 (cumulative sales begin to dwarf forecast, so denominator advantage becomes eroded)
        (pl.col("cumsum_actual_completed_sales") / pl.col("expected_total_sales")).alias("proportion_expected_completed"),
    )

    return product_run_info

"""
Add volatility measures based on forecasted and actual sales history
"""
def __add_volatility_measures(
        product_run_info: pl.DataFrame,
        measure: str = "cv",
    ) -> pl.DataFrame:
        if measure == "std":
            product_run_info = product_run_info.with_columns(
                # Volatility of actual sales so far
                pl.col("actual_completed_sales_weekly").list.std().alias("std_actual_completed_sales"),

                # Volatility of forecasted sales so far
                pl.col("forecasted_remaining_sales_weekly").list.std().alias("std_forecasted_remaining_sales"),
            )
        
        elif measure == "cv":
            product_run_info = product_run_info.with_columns(
                # Standard deviation relative to mean: coefficient of variation (cv)
                (pl.col("forecasted_remaining_sales_weekly").list.std() / pl.col("forecasted_remaining_sales_weekly").list.mean()).alias("cv_forecasted_remaining_sales"),
                (pl.col("actual_completed_sales_weekly").list.std() / pl.col("actual_completed_sales_weekly").list.mean()).alias("cv_actual_completed_sales"),
            )

        else:
            raise ValueError(f"Unknown volatility measure: {measure}")
        
        

        return product_run_info

"""
Add product status features that are dependent on weeks_out features - how long since the product's range_out status last changed, and how many times it has changed
"""
def __add_weekly_product_status(
    product_run_info: pl.DataFrame
) -> pl.DataFrame:
    product_run_info = product_run_info.with_columns(
        pl.when(
            pl.col("n_product_out_flips") > 0
        ).then(
            # Get week of last product out flip
            (
                pl.col("start_runtime") - pl.col("product_out_flips").list.arg_max()
            )
        ).otherwise(
            # No flips - set to start of product run
            pl.col("start_runtime")
        ).alias("last_product_out_flip_time"),
    )

    product_run_info = product_run_info.with_columns(
        # How long this product has retained its current status
        (pl.col("weeks_out").cast(pl.Int8) - pl.col("last_product_out_flip_time").cast(pl.Int8)).abs().alias("weeks_since_last_product_out_flip"),
    )

    # Add rate of flipping
    product_run_info = product_run_info.with_columns(
        # Rate of flipping = number of flips / weeks_in
        (pl.col("n_product_out_flips") / pl.col("weeks_in")).alias("rate_product_out_flips"),
    )
    
    return product_run_info

"""
Add initial info on how the product's range_out status has changed over time
"""
def __add_product_status_info(
        cum_agg_data: pl.DataFrame
    ) -> pl.DataFrame:
        product_run_info = cum_agg_data.with_columns(
            # Get all times the product status has "flipped"
            pl.col("range_out_weekly").list.eval(
                (
                    (pl.element() != pl.element().shift(-1)).fill_null(False)
                ).cast(pl.UInt8)
            ).alias("product_out_flips"),

            # Starting status
            pl.col("range_out_weekly").list[0].alias("first_product_range_out"),    )
        
        # Add product flip info
        product_run_info = product_run_info.with_columns(
            # How many times has the product's range out status changed
            pl.col("product_out_flips").list.sum().alias("n_product_out_flips"),
        )

        return product_run_info

"""
Add initial info on the product's weeks_out history - how long the product has been running for, whether it has a continuous run, etc
"""
def __add_weeks_out_info(
        product_run_info: pl.DataFrame
    ) -> pl.DataFrame:
        product_run_info = product_run_info.with_columns(
            # Number of weeks the product has run for
            pl.col("weeks_out").list.len().alias("total_runtime"),
            pl.col("weeks_out").list[0].alias("start_runtime"),
            
            # Count number of missing weeks in product's runtime
            pl.col("weeks_out").list.eval(
                (pl.element() - pl.element().shift(-1)).cast(pl.UInt8) != 1
            ).list.sum().alias("n_missing_runtimes"),

            # Restore weeks_out to be a single value (latest week only)
            pl.col("weeks_out").list[-1].alias("weeks_out"),
        )
        
        # Get reversed weeks_out - how many weeks the product has been running for up to the current week
        product_run_info = product_run_info.with_columns(
            (pl.col("start_runtime") - pl.col("weeks_out") + 1).alias("weeks_in"),
        )

        return product_run_info


"""
Add info on previous appearances of the product in earlier catalogues as this affects discontinuation status - products that have appeared in many previous catalogues but have consistent non-discontinued status can be discontinued
"""
def __add_previous_run_info(
    data: pl.DataFrame
) -> pl.DataFrame:
    return data.with_columns(     
        # Check in how many other catalogues this product appears in
        ((pl.col("catalogue").n_unique().over("product")).cast(pl.UInt8) - 1).alias("catalogue_count"),
    )

"""
Split data into train_data and eval_data sets
Because we're predicting for an unseen product in an unseen catalogue, we withhold 20% of newest catalogues for eval 
"""
def __train_eval_split(
    data: pl.DataFrame,
    split_col: str = "catalogue_group",
    eval_data_size: float = 0.2
) -> tuple:

    # Get eval catalogue IDs
    catalogues = data[split_col].unique().sort()
    n_eval_catalogues = round(len(catalogues) * eval_data_size)
    eval_catalogues = catalogues.tail(n_eval_catalogues)

    # Split data into train_data and eval_data sets
    train_data = data.filter(~pl.col(split_col).is_in(eval_catalogues))
    eval_data = data.filter(pl.col(split_col).is_in(eval_catalogues))

    return train_data, eval_data

"""
Separate dependent variable from features
"""
def __seperate_dependent_variable(
    data: pl.DataFrame,
    target: str,
) -> tuple:
    return data.drop(target), data.select(pl.col(target))

"""
Select n random products, with a balance of discontinued and not discontinued
"""
def get_sample_products(
    data,
    n: int = 50,
    target_balance: float = 0.5,
    requested_catalogue = "90"
) -> pl.DataFrame:
    # Get balanced sample
    n_positive_samples = round(n * target_balance)
    n_negative_samples = n - n_positive_samples
    
    # TODO refactor to avoid filtering all data
    positive_data = data.filter(pl.col("discontinued") == True)
    negative_data = data.filter(pl.col("discontinued") == False)
    if requested_catalogue:
        positive_data = positive_data.filter(pl.col("catalogue") == requested_catalogue)
        negative_data = negative_data.filter(pl.col("catalogue") == requested_catalogue)

    positive_samples = positive_data.sample(n = n_positive_samples)
    negative_samples = negative_data.sample(n = n_negative_samples)
    
    sample = pl.concat([positive_samples, negative_samples])

    # Fetch all weekly info from the specific product and catalogue pairs that appear in the sample
    merging_cols = ["product", "catalogue"]
    weekly_sample = data.join(
        sample.select(merging_cols),
        on = merging_cols,
        how = "semi",
    )

    return weekly_sample

"""
Group data by product and aggregate weekly info into arrays to produce a product-level dataframe
"""
def get_product_aggregate(
    data, 
) -> pl.DataFrame:
    by_week_columns = [col_name for col_name in data.columns if "week" in col_name]
    product_columns = [col_name for col_name in data.columns if col_name not in by_week_columns]

    return data.group_by(product_columns).agg(by_week_columns)

"""
Get cumulative lists of weekly data for each product up to the current week
e.g. a product with 4 weeks of data will have:
week_out | cum_weeks_out 
4 | [4] 
3 | [4, 3] 
2 | [4, 3, 2] 
1 | [4, 3, 2, 1]        
"""

def get_product_cum_aggregate(
    data: pl.DataFrame,
    cum_cols: list, 
    order_col: str = "weeks_out",
) -> pl.DataFrame:
    
    by_week_columns = [col_name for col_name in data.columns if "week" in col_name and col_name not in cum_cols]
    product_columns = [col_name for col_name in data.columns if col_name not in by_week_columns and col_name not in cum_cols]

    # Per-product 0-based index 
    frame = data.with_columns(
        idx = (pl.col(order_col).cum_count().over(product_columns) - 1).cast(pl.Int32) 
    )

    # Build full per-product lists for each weekly column
    frame = frame.with_columns(
        [pl.col(c).implode().over(product_columns).alias(f"__{c}__full") for c in cum_cols]
    )

    # Slice each full list up to the current row
    frame = frame.with_columns(
        [
            pl.col(f"__{c}__full")
              .list.slice(0, pl.col("idx") + 1)
              .alias(c)
            for c in cum_cols 
        ]
    )

    return frame.select([*product_columns, *by_week_columns, *cum_cols])

"""
Prepare data in train_dataing format - converting booleans, etc
"""
def __transform_for_training(
    data: pl.DataFrame,
    target_col: str,
    group_cols: list,
):
    # Make clear which columns aren't features
    data_rename_map = {
        target_col: f"{target_col}_target",
    }
    
    [data_rename_map.update({col: f"{col}_group"}) for col in group_cols]

    data = data.rename(data_rename_map)

    # Cast all booleans to 0/1 UInt8 integers
    bool_cols = [col for col in data.columns if data[col].dtype == pl.Boolean]
    data = data.with_columns(
        [
            pl.col(col).cast(pl.UInt8).alias(f"{col}_bool")
        for col in bool_cols]
    )

    return data

"""
Get scaling info from training data - min, max, mean, std for each numeric column
"""
def __get_scaling_info(
    data: pl.DataFrame,
    stats: list = ["min", "max", "mean", "std"],
) -> pl.DataFrame:

    num_cols = [col for col in data.columns if data[col].dtype.is_numeric() and "group" not in col and "target" not in col and "bool" not in col]

    # Compute summary statistics
    scaling_info = data.select(
        # Remove non-numeric columns
        [pl.col(c).alias(c) for c in num_cols]
    ).melt(
        # "Melt" dataframe to long format - one row per variable per observation
        variable_name="feature", value_name="value"
    ).group_by(
        # Group by feature to produce list of values per feature
        "feature"
    ).agg(
        # Aggregate to multiple columns for summary statistics
        min_stat = pl.col("value").min(),
        max_stat = pl.col("value").max(),
        mean_stat = pl.col("value").mean(),
        std_stat = pl.col("value").std(),     
    ).sort(
        # Sort by feature name to guarantee repeatability
        "feature"
    )

    return scaling_info 
    
"""
Apply scaling to numeric columns based on provided scaling info
"""
def __apply_scaling(
    data: pl.DataFrame,
    scaling_info: pl.DataFrame,
) -> pl.DataFrame:
    # TODO
    return data

"""
Undo scaling to numeric columns based on provided scaling info
"""
def undo_scaling(
    data: pl.DataFrame,
    scaling_info: pl.DataFrame,
) -> pl.DataFrame:
    # TODO
    return data

"""
Low frequency categorical variables can cause data leakage, e.g. if a product appears once and its supplier is unique to it, the model can learn to associate that supplier with discontinuation
To avoid data leakage and reduce cardinality in categorical columns, form distribution of counts and categorical values, and only keep 95% of rows and bin the rest as 'other/unknown'
"""

def __address_high_cardinality(
    data: pl.DataFrame,
    small_supplier_threshold: float = 0.05,
    other_unknown_label: str = "other_unknown",
) -> pl.DataFrame:
    
    # Do not perform this process on grouping cols
    cat_cols = [col for col in data.columns if data[col].dtype == pl.Categorical and "group" not in col and "target" not in col]
    
    for col in cat_cols:
        value_counts = data[col].value_counts().sort("count", descending = True)
        total_count = value_counts["count"].sum()
        value_counts = value_counts.with_columns(
            (pl.col("count") / total_count).alias("proportion")
        )

        # Take cumulative sum of proportions, from highest to lowest
        value_counts = value_counts.with_columns(
            pl.col("proportion").cum_sum().alias("cumulative_proportion")
        )

        # Keep right hand side of values distribution - i.e. those that make up (1 - m)% of data
        values_to_keep = value_counts.filter(pl.col("cumulative_proportion") <= (1 - small_supplier_threshold))[col].to_list()
        
        # Replace values not in values_to_keep with other_unknown_label
        keep_category = pl.col(col).cast(pl.Utf8).is_in(values_to_keep)

        data = data.with_columns(
            pl.when(keep_category)
              .then(pl.col(col))
              .otherwise(pl.lit(other_unknown_label))
              .cast(pl.Categorical)
              .alias(col)
        )

    return data

"""
Basic tests on test/train to ensure they meet expectations
"""
def __test_data(
    data: pl.DataFrame,
    data_name: str,
    group_cols: list,
):
    # Check if polars
    if not isinstance(data, pl.DataFrame):
        raise ValueError(f"{data_name} is not a polars DataFrame")

    # Check if empty
    if data.is_empty():
        raise ValueError(f"{data_name} is empty")

    # Locate null rows
    data_missing = data.filter(
        pl.any_horizontal(
            pl.all().is_null()
        )
    )

    if not data_missing.is_empty():
        # For each null row, get which columns are null
        data_missing = data_missing.with_columns( 
            cols_with_missing = pl.concat_list(
                *[
                    pl.when(
                        pl.col(col).is_null()
                    ).then(pl.lit(col)).otherwise(pl.lit(None))
                for col in data.columns]
            ).alias("cols_with_missing")
        )

        # Get identifying info for each missing row
        identifying_info = data_missing.select(
            pl.col(group_cols).cast(pl.Utf8)
            ).sort(group_cols[::-1])

        n_missing = data_missing.height
        missing_cols = data_missing.select(
            pl.col("cols_with_missing").explode().unique().drop_nulls()
        ).to_series().to_list()

        raise ValueError(f"{data_name} has {n_missing} rows with missing values \n\n Columns containing NaN/null: {missing_cols} \n\n Row indexes containin NaN/null: {identifying_info}")








