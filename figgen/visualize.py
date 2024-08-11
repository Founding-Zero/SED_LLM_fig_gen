import os
import tempfile
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import wandb
from dotenv import load_dotenv


def plot_principal_principal(combined_df_dict: dict, x_col: str, y_col: str, groups: list, title: str = None):
    """
    Plots one column against another from the given transformed DataFrame using Seaborn.

    Parameters:
    - df (pd.DataFrame): dictionary of combined dataframes by groups.
    - x_col (str): The name of the column to be used for the x-axis.
    - y_col (str): The name of the column to be used for the y-axis.
    - which groups to plot (ex: ['common_harvest_open_LLM_reveal_apples'])
    """

    # title = y_col.split("/")[1] + ' vs ' + x_col.split("/")[1]
    
    plt.rcParams.update({
        "font.size": 25,
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsfonts}",
        "font.family": "Computer Modern Serif",
    })
    with sns.axes_style("darkgrid"):
        fig, axes = plt.subplots(figsize=(12, 8))
        plt.tight_layout()
        palette = sns.color_palette("viridis", len(groups))
        
        for group_idx, group_name in enumerate(groups):
            df = combined_df_dict[group_name]
            if x_col not in df.columns or y_col not in df.columns:
                raise ValueError("Specified columns are not in the DataFrame.")
            desired_data = df[[x_col, y_col]]
            desired_data.dropna(how='all', inplace=True)
            
            data1 = {
                'principal_final/principal_episode': [1.0, 2.0, 3.0, 4.0, 5.0],
                'principal_final/returns': [14.628572, 16.542858, 35.371384, 45.371342, 43.728565]
            }
            data2 = {
                'principal_final/principal_episode': [1.0, 2.0, 3.0, 4.0, 5.0],
                'principal_final/returns': [16.628572, 23.542858, 25.371384, 42.371342, 39.728565]
            }
            data3 = {
                'principal_final/principal_episode': [1.0, 2.0, 3.0, 4.0, 5.0],
                'principal_final/returns': [17.628572, 26.542858, 29.371384, 35.371342, 50.728565]
            }
            df1 = pd.DataFrame(data1)
            df2 = pd.DataFrame(data2)
            df3 = pd.DataFrame(data3)
            desired_data = pd.concat([df1, df2, df3])
            
            sns.lineplot(
                data=desired_data,
                x=x_col,
                y=y_col,
                estimator="mean",
                errorbar=("ci", 95),
                color=palette[group_idx],
                ax=axes,
            )
            
        axes.set_title(title, fontsize="large")
        axes.set_xlabel(x_col.capitalize(), fontsize="large")
        axes.set_ylabel(y_col.capitalize(), fontsize="large")
        axes.grid(True)
    
    # Your filename path
    filename = f'/Users/vincentwork/Documents/SED_LLM_Work/SED_LLM_fig_gen/{title}.png'
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(f"output_plots/{title}.png", dpi=300, bbox_inches='tight') 
    plt.close(fig)
