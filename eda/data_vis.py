"""
Functions to show different visualisations of the data
"""

import polars as pl

from matplotlib import pyplot as plt
import numpy as np

from src.data.data_prep import get_sample_products, get_product_aggregate 

"""
Shows all visualisations
"""
def show_all_vis(data):
    # Data is split in preparation for training, rejoin for visualisation 
    data = data[0].with_columns(
        discontinued_target_bool = data[1]["discontinued_target_bool"]
    )
    
    # Remove nuisance effects
    # Only observe one catalogue
    data = data.filter(pl.col("catalogue_group") == "90")

    # Only observe products appearing for first time
    data = data.filter(pl.col("catalogue_count") == 0)

    __show_product_growth_progression(data, n_shows = 3, n_products_per_show = 4)
    
    # Status
    # Show 'easy cases': discontinued == range_out_weekly, at weeks_out = 1 and n_product_flips = 0

    # Show lead times effect: product categories (supplier, broad_category, specific_category) with last_product_out_flip_time, at weeks_out = 1 and n_product_flips = 1
    
    # Growth
    # Show growth and discontinued

    # Show volatility changers with cv

"""
Show product growth progression for a sample of products
"""
def __show_product_growth_progression(
    data: pl.DataFrame,
    n_shows: int,
    n_products_per_show: int,
):
    for _ in range(n_shows):
        sample = get_sample_products(data, n = n_products_per_show)
        agg_sample = get_product_aggregate(sample)
    
        __plot_aggregate(agg_sample, "weeks_out", "log_estimated_growth_sales_weekly")

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


