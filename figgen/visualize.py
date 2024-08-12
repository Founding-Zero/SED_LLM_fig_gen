import os
import tempfile
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import wandb
from dotenv import load_dotenv
from scipy import stats

def analyze_threshold(df, threshold, return_col, step_column, confidence=0.95):
    # Find the first step that surpasses the threshold
    steps_to_threshold = df[df[return_col] >= threshold][step_column].min()
    
    # Calculate mean and standard error
    mean = steps_to_threshold
    std_error = df[df[step_column] <= steps_to_threshold][return_col].sem()
    
    # Calculate confidence interval
    ci = stats.t.interval(confidence, len(df) - 1, loc=mean, scale=std_error)
    
    return mean, ci

def plot_results(thresholds, means, cis, group_name):
    # Create a DataFrame for seaborn
    plot_df = pd.DataFrame({
        'Threshold': thresholds,
        'Average Steps': means,
        'CI_lower': [ci[0] for ci in cis],
        'CI_upper': [ci[1] for ci in cis]
    })
    
    # Set the style and color palette
    sns.set_style("whitegrid")
    sns.set_palette("deep")
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x='Threshold', y='Average Steps', data=plot_df, capsize=0.1)
    
    # Add error bars
    ax.errorbar(x=range(len(thresholds)), y=means, 
                yerr=[means - plot_df['CI_lower'], plot_df['CI_upper'] - means],
                fmt='none', c='black', capsize=5)
    
    title = group_name.split('_vin3_')[1]
    # Customize the plot
    ax.set_title(f"Average Steps to Reach Threshold: {title}", fontsize=16)
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Average Steps', fontsize=12)
    
    # Add value labels on the bars
    for i, v in enumerate(means):
        ax.text(i, v, f'{v:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"output_plots/average_time_to_good_{group_name}.png", dpi=300, bbox_inches='tight')


def estimate_time_to_threshold(combined_df_dict: dict, x_col: str, y_col: str, groups_list: list, title: str = None):
    outer_groups = ['LLM', 'Random', 'Dual-RL', 'AID'] 
    outer_groups_dict = {
        'LLM': [],
        'Random': [],
        'Dual-RL': [],
        'AID': [],
    }
    for group in groups_list:
        for outer_group in outer_groups:
            if outer_group in group:
                outer_groups_dict[outer_group].append(group)
    
    # Remove duplicates
    for outer_group, groups in outer_groups_dict.items():
        outer_groups_dict[outer_group] = list(set(groups))
    
    # for parent_group, groups in outer_groups_dict.items():  
    for paranet_group, groups in outer_groups_dict.items():
        for group_idx, group_name in enumerate(groups):
            df = combined_df_dict[group_name]
            if  "LLM" in group_name or "Random" in group_name:
                x_col = "combined_val_train/episode"
                y_col = "principal_final/returns"
                df[x_col] = df[x_col] * 1000
            elif "AID" in group_name:
                x_col = "validation/episode"
                y_col = "validation/game 0 mean return"
                df[x_col] = df[x_col] * 2000
            elif "Dual-RL" in group_name:
                x_col = "principal_final/principal_step"
                y_col = "principal_final//game 0 principal return"
                df[x_col] = df[x_col] * 20000 
        
                
            # thresholds = [40, 45, 50]  # Replace with your desired thresholds
            # means = []
            # cis = []

            # for threshold in thresholds:
            #     mean, ci = analyze_threshold(df, threshold, y_col, x_col)
            #     means.append(mean)
            #     cis.append(ci)

            # plot_results(thresholds, means, cis, group_name)
            # Assuming your dataframe is called 'df' and has columns for each run and a 'step' column
            thresholds = [40, 45, 50]
            results = {threshold: [] for threshold in thresholds}

            for column in df.columns:
                if column != 'step':  # Skip the step column
                    run_data = df[column]
                    for threshold in thresholds:
                        # Find the first step where the value exceeds the threshold
                        steps_to_threshold = np.argmax(run_data > threshold)
                        if steps_to_threshold == 0 and run_data.max() <= threshold:
                            # Threshold never reached, use max steps or exclude
                            steps_to_threshold = len(run_data)  # or np.nan to exclude
                        results[threshold].append(steps_to_threshold)

            # Calculate averages
            averages = {threshold: np.mean(steps) for threshold, steps in results.items()}

            for threshold, avg_steps in averages.items():
                print(f"{group_name} Average steps to reach threshold {threshold}: {avg_steps:.2f}")


def plot_principal_principal_line(combined_df_dict: dict, x_col: str, y_col: str, groups_list: list, title: str = None):
    """
    Plots one column against another from the given transformed DataFrame using Seaborn.

    Parameters:
    - df (pd.DataFrame): dictionary of combined dataframes by groups.
    - x_col (str): The name of the column to be used for the x-axis.
    - y_col (str): The name of the column to be used for the y-axis.
    - which groups to plot (ex: ['common_harvest_open_LLM_reveal_apples'])
    """    
    plt.rcParams.update({
        "font.size": 25,
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsfonts}",
        "font.family": "Computer Modern Serif",
    })
    
    outer_groups = ['LLM', 'Random', 'Dual-RL', 'AID'] 
    outer_groups_dict = {
        'LLM': [],
        'Random': [],
        'Dual-RL': [],
        'AID': [],
    }
    for group in groups_list:
        for outer_group in outer_groups:
            if outer_group in group:
                outer_groups_dict[outer_group].append(group)
    
    # Remove duplicates
    for outer_group, groups in outer_groups_dict.items():
        outer_groups_dict[outer_group] = list(set(groups))
        
    
    for parent_group, groups in outer_groups_dict.items():  
        if parent_group != 'LLM':
            continue 
        with sns.axes_style("darkgrid"):
            fig, axes = plt.subplots(figsize=(12, 8))
            plt.tight_layout()
            # fig.subplots_adjust(left=0.05, right=0.85, top=0.85, bottom=0.15, wspace=0.2)
            palette = sns.color_palette("viridis", len(groups))
            # palette = sns.color_palette("mako", len(groups))
            
            for group_idx, group_name in enumerate(groups):
                df = combined_df_dict[group_name]
                if  "LLM" in group_name or "Random" in group_name:
                    x_col = "combined_val_train/episode"
                    y_col = "principal_final/returns"
                    df = df[df[x_col] % 20 == 0] # 20 combined steps per principal step
                    df[x_col] = df[x_col] / 20
                    # df[x_col] = df[x_col] * 1000
                    Hparams = "Temperatures" if "LLM" in group_name else "Seed"
                elif "AID" in group_name:
                    x_col = "validation/episode"
                    y_col = "validation/game 0 mean return"
                    df[x_col] = df[x_col] * 2000
                    # df = df[df[x_col] % 20 == 0]
                    # df[x_col] = df[x_col] / 20
                    Hparams = "Princiapl LR"
                elif "Dual-RL" in group_name:
                    x_col = "principal_final/principal_step"
                    y_col = "principal_final//game 0 principal return"
                    df[x_col] = df[x_col] * 20000 
                    Hparams = "Princiapl LR"
                    
                if x_col not in df.columns or y_col not in df.columns:
                    raise ValueError("Specified columns are not in the DataFrame.")
                desired_data = df[[x_col, y_col]]
                desired_data.dropna(how='all', inplace=True)

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
                # axes.legend(title=Hparams, bbox_to_anchor=(1.05, 1), loc='upper left')
                axes.legend(title=Hparams, fontsize='small')
                axes.grid(True)
        
        # # Your filename path
        # filename = f'/Users/vincentwork/Documents/SED_LLM_Work/SED_LLM_fig_gen/{group_name}.png'
        # # Ensure the directory exists
        # os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(f"output_plots/{parent_group}.png", dpi=300, bbox_inches='tight') 
        plt.close(fig)

    
    # with sns.axes_style("darkgrid"):
    #     fig, axes = plt.subplots(figsize=(12, 8))
    #     plt.tight_layout()
    #     # fig.subplots_adjust(left=0.05, right=0.85, top=0.85, bottom=0.15, wspace=0.2)
    #     palette = sns.color_palette("viridis", len(groups))
    #     # palette = sns.color_palette("flare", len(groups))
            
    #     for parent_group, groups in outer_groups_dict.items():
    #         for group_idx, group_name in enumerate(groups):
    #             df = combined_df_dict[group_name]
    #             if  "LLM" in group_name or "Random" in group_name:
    #                 x_col = "combined_val_train/episode"
    #                 y_col = "principal_final/returns"
    #                 df = df[df[x_col] % 20 == 0] # 20 combined steps per principal step
    #                 df[x_col] = df[x_col] / 20
    #                 Hparams = "Temperatures" if "LLM" in group_name else "Seed"
    #             elif "AID" in group_name:
    #                 x_col = "validation/episode"
    #                 y_col = "validation/game 0 mean return"
    #                 df[x_col] = df[x_col] * 2
    #                 df = df[df[x_col] % 20 == 0]
    #                 df[x_col] = df[x_col] / 20
    #                 Hparams = "Princiapl LR"
    #             elif "Dual-RL" in group_name:
    #                 x_col = "principal_final/principal_step"
    #                 y_col = "principal_final//game 0 principal return"
    #                 # df[x_col] = df[x_col] * 20 
    #                 Hparams = "Princiapl LR"
                    
    #             if x_col not in df.columns or y_col not in df.columns:
    #                 raise ValueError("Specified columns are not in the DataFrame.")
    #             desired_data = df[[x_col, y_col]]
    #             desired_data.dropna(how='all', inplace=True)

    #             # Extracting label_processed as a tuple
    #             label_processed = group_name.split("_ps_")[-1] if parent_group == "LLM" else group_name.split("_")[-1]

    #             if parent_group == "AID" or parent_group == "Dual-RL":
    #                 label_processed = (group_name.split("_")[-1],)
    #                 # Convert each element in the tuple to scientific notation
    #                 formatted_labels = [f"{float(label):.1e}" for label in label_processed]
    #                 # Create a single label string if needed
    #                 label_string = ", ".join(formatted_labels)
    #             else:
    #                 label_string = label_processed
                
    #             sns.lineplot(
    #                 data=desired_data,
    #                 x=x_col,
    #                 y=y_col,
    #                 estimator="mean",
    #                 errorbar=("ci", 95),
    #                 color=palette[group_idx],
    #                 label=parent_group,
    #                 ax=axes,
    #             )
                
    #             axes.set_title(f"Hyperparameter Comparison - Method: {parent_group}", fontsize="large")
    #             axes.set_xlabel(x_col.capitalize(), fontsize="large")
    #             axes.set_ylabel(y_col.capitalize(), fontsize="large")
    #             axes.set_xlabel("Environment Step")
    #             axes.set_ylabel("Rewards")
    #             if parent_group == "LLM":
    #                 axes.set_xlim(0, 25)
    #             if parent_group == "Random":
    #                 axes.set_xlim(0, 105)
    #             if parent_group == "AID":
    #                 axes.set_xlim(0, 510)
    #             if parent_group == "Dual-RL":
    #                 axes.set_xlim(0, 610)
    #             # axes.legend(title=Hparams, bbox_to_anchor=(1.05, 1), loc='upper left')
    #             axes.legend(title=Hparams, fontsize='small')
    #             axes.grid(True)
