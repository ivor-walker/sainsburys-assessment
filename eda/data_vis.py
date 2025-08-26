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
        discontinued = data[1].alias("discontinued"),
    )

    while True:
        sample = get_sample_products(data, n = 4) 
        agg_sample = get_product_aggregate(sample)
    
    # __plot_aggregate(agg_sample, "weeks_out", "forecasted_remaining_sales_weekly")
    # __plot_aggregate(agg_sample, "weeks_out", "actual_completed_sales_weekly")
        __plot_aggregate(agg_sample, "weeks_out", "forecasted_remaining_revenue_weekly")
        __plot_aggregate(agg_sample, "weeks_out", "actual_completed_revenue_weekly")
        plt.close('all')

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
    line_colour_col: str = "discontinued",
    point_colour_col: str = "range_out_weekly",
):
    fig, axs = plt.subplots(1, 1, figsize=(10, 6))
    for i, row in enumerate(data.iter_rows(named = True)):
        highlight_rgb = min(selected_rgb + (i / len(data)) * gradient_rgb, 1)
        point_colours = [__create_rgb(
            point_colour_true,
            highlight_rgb,
        ) for point_colour_true in row[point_colour_col]]

        line_colour = __create_rgb(
            row[line_colour_col],
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
            label = f"{row['product']} / {row['catalogue']} / {row['broad_category']} / {row['domestic']} / {row['seasonal']}"
        )

    if invert_x_axis:
        axs.invert_xaxis()

    axs.set_xlabel(x)
    axs.set_ylabel(y)
    axs.set_title(f"{y} vs {x} by {line_colour_col} and {point_colour_col}")
    axs.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small', title = "Product / Catalogue / Broad Category / Domestic / Seasonal")
    
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


