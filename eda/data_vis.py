"""
Functions to show different visualisations of the data
"""

import polars as pl

from matplotlib import pyplot as plt
import numpy as np

from src.data.data_prep import get_sample_products, get_product_aggregate 

from src.data.data_prep import undo_scaling

"""
Shows all visualisations
"""
def show_all_vis(data, scaling_info):
    # Data is split in preparation for training, rejoin for visualisation 
    data = data[0].with_columns(
        discontinued_target_bool = data[1]["discontinued_target_bool"]
    )

    data = undo_scaling(data, scaling_info)

    # Remove nuisance effects
    # Only observe products appearing for first time (10% of products and rows)
    data = data.filter(pl.col("catalogue_count") == 0)

    # Observe products at the end of their run only
    data = data.filter(pl.col("weeks_out") == 1)

    # Growth
    # Show growth and discontinued
    # __show_sample_products(data, n_shows = 3)
    __show_seperation(data,
        x = "discontinued_target_bool",
        y = "log_estimated_growth_sales_weekly",
    )

    # Show volatility changers with cv
    volatility_changers = ["domestic_bool", "seasonal_bool"]
    for changer in volatility_changers:
        __show_seperation(data,
            x = changer,
            y = "cv_forecasted_remaining_sales",
        )

    # Status
    # TODO improve visualisation for this
    # Show 'easy cases': discontinued == range_out_weekly, at weeks_out = 1 and n_product_out_flips = 0
    easy_cases = data.filter(
        (pl.col("n_product_out_flips") == 0) &
        (pl.col("discontinued_target_bool") == pl.col("range_out_weekly_bool"))
    )
    # __show_sample_products(easy_cases, n_shows = 3)
    __show_rates(data,
        y = "discontinued_target_bool",
        x = "range_out_weekly_bool"
    )
    __show_rates(easy_cases,
        y = "discontinued_target_bool",
        x = "range_out_weekly_bool",
    )

    # Show lead times effect: product categories (supplier, broad_category, specific_category) with last_product_out_flip_time, at weeks_out = 1 and n_product_out_flips = 1
    lead_time_cases = data.filter(
        (pl.col("weeks_out") == 1) &
        (pl.col("n_product_out_flips") == 1)
    )
    product_category_types = ["supplier", "broad_category", "specific_category"]
    product_categories = {
        category: [
            col for col in lead_time_cases.columns if category in col
        ]
    for category in product_category_types}
    
    for category in product_category_types:
        __show_dist(lead_time_cases,  
            x = product_categories[category],
            y = "discontinued_target_bool",
        )
    
"""
Show for a sample of products
"""
def __show_sample_products(
    data: pl.DataFrame,
    n_shows: int,
    n_products_per_show: int = 4,
    x: str = "weeks_out",
    y: str = "log_estimated_growth_sales_weekly",
):
    for _ in range(n_shows):
        sample = get_sample_products(data, n = n_products_per_show)
        agg_sample = get_product_aggregate(sample)
    
        __plot_aggregate(agg_sample, x, y)

        plt.close("all")

"""
Plot aggregated data, coloured by discontinued status
"""
def __plot_aggregate(
    data,
    x: str,
    y: str,
    invert_x_axis: bool = True,
    selected_rgb: float = 0.4,
    gradient_rgb: float = 0.8,
    line_colour_col: str = "discontinued_target_bool",
    point_colour_col: str = "range_out_weekly_bool",
):
    fig, axs = plt.subplots(1, 1, figsize=(10, 6))
    for i, row in enumerate(data.iter_rows(named = True)):
        highlight_rgb = min(selected_rgb + (i / len(data)) * gradient_rgb, 1)
        point_colours = [__create_rgb(
            point_colour_true == 1,
            highlight_rgb,
        ) for point_colour_true in row[point_colour_col]]

        line_colour = __create_rgb(
            row[line_colour_col] == 1,
            highlight_rgb,
        )

        
        for index, (x_point, y_point) in enumerate(zip(row[x], row[y])):
            axs.plot(
                x_point,
                y_point,
                color = point_colours[index],
                marker = "o",
                linestyle = 'None',
                alpha = 1,
            )
        
        axs.plot(
            row[x],
            row[y],
            color = line_colour,
            linestyle = '-',
            linewidth = 2,
            alpha = 0.5,
            label = f"{row['product_group']} / {row['catalogue_group']} / {row['domestic_bool']} / {row['seasonal_bool']}"
        )

    if invert_x_axis:
        axs.invert_xaxis()

    axs.set_xlabel(x)
    axs.set_ylabel(y)
    axs.set_title(f"{y} vs {x} by {line_colour_col} and {point_colour_col}")
    axs.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small', title = "Product / Catalogue / Domestic / Seasonal")
    
    plt.tight_layout()
    plt.show()

"""
Helper function to create RGBs
"""
def __create_rgb(
    negative: bool,
    highlight_rgb: float,
    base_rgb: float = 0.2,
) -> list:
    if negative:
        return [base_rgb, highlight_rgb, base_rgb]
    
    else:

        return [highlight_rgb, base_rgb, base_rgb]

"""
Show box (stem and whiskers) plot of mean y for each x category
"""
def __show_seperation(data,
    x: str,
    y: str,
):
    mean_y_str = f"mean_{y}"
    mean_data = data.group_by(x).agg(
        pl.count().alias("count"),
        pl.col(y).mean().alias(mean_y_str),
        pl.col(y).std().alias(f"std_{y}"),
    ).sort(x)

    fig, axs = plt.subplots(1, 1, figsize=(8, 6))

    axs.boxplot(
        [data.filter(pl.col(x) == category)[y].to_list() for category in mean_data[x]],
        labels = mean_data[x].to_list(),
        showmeans = True,
        meanline = True,
        patch_artist = True,
        boxprops = dict(facecolor="lightblue", color="blue"),
        medianprops = dict(color="red"),
        meanprops = dict(color="green"),
        showfliers = False,
    )

    axs.set_xlabel(x)
    axs.set_ylabel(y)

    axs.set_title(f"{y} distribution by {x}")

    plt.tight_layout()
    plt.show()

"""
Show mean of some boolean variable y for each boolean variable x
"""
def __show_rates(data,
    y: str,
    x: str,
):
    mean_y_str = f"mean_{y}"
    mean_data = data.group_by(x).agg(
        pl.count().alias("count"),
        pl.col(y).mean().alias(mean_y_str),
        pl.col(y).std().alias(f"std_{y}"),
    ).sort(x)

    fig, axs = plt.subplots(1, 1, figsize=(8, 6))

    axs.bar(
        mean_data[x].to_list(),
        mean_data[mean_y_str].to_list(),
        color = ["lightblue", "lightgreen"],
        edgecolor = "black",
        capsize = 10,
    )

    axs.set_xlabel(x)
    axs.set_ylabel(f"Mean {y}")

    axs.set_title(f"Mean {y} by {x}")

    plt.tight_layout()
    plt.show()

# TODO
"""
Show distribution of some boolean variable y for each category in x
"""
def __show_dist(data,
    x: list[str],
    y: str,
):
    pass 
