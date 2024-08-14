import os
import tempfile
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import wandb
from dotenv import load_dotenv
from scipy import stats
from figgen import Config

def format_number(num):
    if abs(num) >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif abs(num) >= 1_000:
        return f"{num/1_000:.1f}k"
    else:
        return str(num)
    

def plot_principal_principal(combined_df_dict: dict, groups_list: list, cfg: Config):
    """
    Outputs 1 plot, 1 for each method to the output_plots directory.

    Parameters:
    - df (pd.DataFrame): dictionary of combined dataframes by groups.
    - groups_list (list(str)): which groups to plot (ex: ['common_harvest_open_LLM_reveal_apples', 'common_harvest_open_aid_reveal_apples'])
    - cfg (Config): configuration object containing the axis and metrics to plot.
    """    
    plt.rcParams.update({
        "font.size": 25,
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsfonts}",
        "font.family": "Computer Modern Serif",
    })
    
    outer_groups = cfg.methods
    outer_groups_dict = defaultdict(list)
    for group in groups_list:
        for outer_group in outer_groups:
            if outer_group in group:
                outer_groups_dict[outer_group].append(group)
    
    # Remove duplicates
    for outer_group, groups in outer_groups_dict.items():
        outer_groups_dict[outer_group] = list(set(groups))
        
    
    for parent_group, groups in outer_groups_dict.items():  
        with sns.axes_style("darkgrid"):
            fig, axes = plt.subplots(figsize=(12, 8))
            plt.tight_layout()
            # fig.subplots_adjust(left=0.05, right=0.85, top=0.85, bottom=0.15, wspace=0.2)
            # palette = sns.color_palette("viridis", len(groups))
            palette = cfg.custom_palette
            
            for group_idx, group_name in enumerate(tqdm(groups, desc=f"Processing {parent_group} data")):
                df = combined_df_dict[group_name]
                df.rename(columns={"principal_final/principal_step": cfg.axis}, inplace=True)
                x_col = cfg.axis
                if  "LLM" in group_name or "Random" in group_name:
                    y_col = cfg.metrics_by_method[0]
                    Hparams = "Prompt_Styles" if "LLM" in group_name else "Seed"
                elif "AID" in group_name or "Dual-RL" in group_name:
                    y_col = cfg.metrics_by_method[1]
                    Hparams = "Princiapl LR"
                    
                if x_col not in df.columns or y_col not in df.columns:
                    raise ValueError("Specified columns are not in the DataFrame.")
                desired_data = df[[x_col, y_col]]
                desired_data.dropna(how='all', inplace=True)
                desired_data[x_col] *= 1000 # Convert to environment steps
                
                # Extracting label_processed as a tuple
                label_processed = group_name.split("_ps_")[-1] if parent_group == "LLM" else group_name.split("_")[-1]

                if parent_group == "AID" or parent_group == "Dual-RL":
                    label_processed = (group_name.split("_")[-1],)
                    # Convert each element in the tuple to scientific notation
                    formatted_labels = [f"{float(label):.1e}" for label in label_processed]
                    # Create a single label string if needed
                    label_string = ", ".join(formatted_labels)
                else:
                    label_string = label_processed
                
                formatted_x_ticks = [format_number(x_tick) for x_tick in desired_data[x_col].unique()]
                
                sns.lineplot(
                    data=desired_data,
                    x=x_col,
                    y=y_col,
                    estimator="mean",
                    errorbar=("ci", 95),
                    color=palette[group_idx],
                    label=label_string,
                    ax=axes,
                )
                
            axes.set_title(f"Hyperparameter Comparison - Method: {parent_group}", fontsize="large")
            axes.set_xlabel(x_col.capitalize(), fontsize="large")
            axes.set_ylabel(y_col.capitalize(), fontsize="large")
            axes.set_xlabel("Environment Steps")
            axes.set_ylabel("Rewards")
            if parent_group == "LLM":
                axes.set_xlim(0, )
            if parent_group == "Random":
                axes.set_xlim(0, )
            if parent_group == "AID":
                axes.set_xlim(0, )
            if parent_group == "Dual-RL":
                axes.set_xlim(0, )
            # axes.set_xticks(desired_data[x_col])
            # axes.set_xticklabels(formatted_x_ticks, rotation=90)
            # axes.legend(title=Hparams, bbox_to_anchor=(1.05, 1), loc='upper left')
            axes.legend(title=Hparams, fontsize='small')
            axes.grid(True)
        
        plt.savefig(f"output_plots/{parent_group}.png", dpi=300, bbox_inches='tight') 
        plt.close(fig)