from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from figgen import DataAnalyzer

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

            # for group_idx, group_name in enumerate(groups):
            #     df = combined_df_dict[group_name]
            #     if  "LLM" in group_name or "Random" in group_name:
            #         x_col = "combined_val_train/episode"
            #         y_col = "principal_final/returns"
            #         df = df[df[x_col] % 20 == 0] # 20 combined steps per principal step
            #         df[x_col] = df[x_col] / 20
            #         # df[x_col] = df[x_col] * 1000
            #         Hparams = "Temperatures" if "LLM" in group_name else "Seed"
            #     elif "AID" in group_name:
            #         x_col = "validation/episode"
            #         y_col = "validation/game 0 mean return"
            #         df[x_col] = df[x_col] * 2000
            #         # df = df[df[x_col] % 20 == 0]
            #         # df[x_col] = df[x_col] / 20
            #         Hparams = "Princiapl LR"
            #     elif "Dual-RL" in group_name:
            #         x_col = "principal_final/principal_step"
            #         y_col = "principal_final//game 0 principal return"
            #         df[x_col] = df[x_col] * 20000 
            #         Hparams = "Princiapl LR"
                
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

            
                # def pull_run_data(self, cfg: Config):
                #     self.set_runs(run_ids=cfg.run_ids)
                #     # temp_runs_list = []
                #     # for run in self.runs:
                #     #     if int(run.name.split("-")[-1]) > 126:
                #     #         temp_runs_list.append(run)
                #     # self.runs = temp_runs_list
                #     # self.set_histories()
            
            # history_list = list(self.histories[run.id])
            # history_df = pd.DataFrame(history_list)
            # # relevant_headers = []
            # # prefixes = cfg.prefixes
            # # for column in history_df.columns:
            # #     if column.startswith(tuple(prefixes)):
            # #         relevant_headers.append(column)
            # csv_file_path = f'figgen/pulled_data/{run.name}/csv_dataframe.csv'
            # # history_df[relevant_headers].to_csv(csv_file_path, index=False)
            # history_df.to_csv(csv_file_path, index=True)
            # # self.csv_paths.append(csv_file_path)
            
            
            # if config_json['principal'] == 'LLM' or config_json['principal'] == 'Random':
            #     df.rename(columns={f'{run} - principal_final/returns': 'principal_final/returns'}, inplace=True)
            # if config_json['principal'] == 'AID':
            #     df.rename(columns={f'{run} - validation/game 0 mean return': 'validation/game 0 mean return'}, inplace=True)
            # if config_json['principal'] == 'Dual-RL':
            #     df.rename(columns={f'{run} - principal_final//game 0 principal return': 'principal_final//game 0 principal return'}, inplace=True)

# title = y_col.split("/")[1] + ' vs ' + x_col.split("/")[1]

# data1 = {
#     'principal_final/principal_episode': [1.0, 2.0, 3.0, 4.0, 5.0],
#     'principal_final/returns': [14.628572, 16.542858, 35.371384, 45.371342, 43.728565]
# }
# data2 = {
#     'principal_final/principal_episode': [1.0, 2.0, 3.0, 4.0, 5.0],
#     'principal_final/returns': [16.628572, 23.542858, 25.371384, 42.371342, 39.728565]
# }
# data3 = {
#     'principal_final/principal_episode': [1.0, 2.0, 3.0, 4.0, 5.0],
#     'principal_final/returns': [17.628572, 26.542858, 29.371384, 35.371342, 50.728565]
# }
# df1 = pd.DataFrame(data1)
# df2 = pd.DataFrame(data2)
# df3 = pd.DataFrame(data3)
# desired_data = pd.concat([df1, df2, df3])
class BenchMARLDataAnalyzer(DataAnalyzer):
    def fetch_and_process_sigma_data(self, data_header):
        self.get_runs()
        self.get_histories()

        desired_group = defaultdict(list)
        for run in self.runs:
            if (
                data_header in self.histories[run.id].columns
                and len(self.histories[run.id][data_header]) > self.min_length
            ):
                desired_group[run.config["task_config"]["sigma_vals"]].append(
                    self.histories[run.id][data_header]
                    .iloc[1 : self.min_length]
                    .tolist()
                )

        desired_data = {}
        for sigma, runs in desired_group.items():
            transposed_runs = list(zip(*runs))
            desired_data[sigma] = transposed_runs

        records = []
        for sigma, episode in desired_data.items():
            for episode_index, values in enumerate(episode):
                for value in values:
                    records.append(
                        {"Episode": episode_index, "Sigma": sigma, data_header: value}
                    )
        return pd.DataFrame(records)

    def visualize_all_sigma_data(self, data, title):
        sigma_groups = data["Sigma"].unique()
        palette = sns.color_palette(self.color_scheme, n_colors=len(sigma_groups))
        color_dict = {sigma: color for sigma, color in zip(sigma_groups, palette)}
        with sns.axes_style("darkgrid"):
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.lineplot(
                data=data,
                x="Episode",
                y=f"{title}",
                hue="Sigma",
                dashes=False,
                palette=color_dict,
                err_style="band",
                errorbar="se",
            )
            plt.title(f"{title} across all Sigma groups")
            plt.xlabel("Episodes")
            plt.ylabel(f"{title}")
            plt.xlim(left=0, right=self.min_length)
            plt.ylim(bottom=0)
            plt.grid(True)
            if self.export_to_wandb:
                self.send_to_wandb(fig, title)
            else:
                plt.show()

    def visualize_individual_sigma_data(self, data, title):
        sigma_groups = data["Sigma"].unique()
        palette = sns.color_palette(self.color_scheme, n_colors=len(sigma_groups))
        color_dict = {sigma: color for sigma, color in zip(sigma_groups, palette)}
        self.visualize_all_sigma_data(data, title)

        with sns.axes_style("darkgrid"):
            # Iterate through each sigma group to create individual plots
            for highlighted_sigma in sigma_groups:
                fig, ax = plt.subplots(figsize=(12, 8))
                # Plot each sigma group
                for sigma in sigma_groups:
                    subset = data[data["Sigma"] == sigma]
                    if sigma == highlighted_sigma:
                        # Highlight the selected sigma group
                        sns.lineplot(
                            data=subset,
                            x="Episode",
                            y=f"{title}",
                            color=color_dict[sigma],
                            label=f"{sigma}",
                            linewidth=2.5,
                            errorbar="se",
                        )
                    else:
                        # Dim other sigma groups
                        sns.lineplot(
                            data=subset,
                            x="Episode",
                            y=f"{title}",
                            color=color_dict[sigma],
                            label=f"{sigma}",
                            linewidth=1,
                            errorbar=None,
                            alpha=0.4,
                        )

                plt.title(f"{title} (Sigma {highlighted_sigma} Highlighted)")
                plt.xlabel("Episodes")
                plt.ylabel(f"{title}")
                plt.xlim(left=0, right=self.min_length)
                plt.ylim(bottom=0)
                plt.legend(title="Sigma")
                plt.grid(True)
                if self.export_to_wandb:
                    self.send_to_wandb(fig, title)
                else:
                    plt.show()

    def plot_all_sigma_data(self):
        pertinent_headers = [
            "collection/agents/reward/episode_reward_min",
            "collection/agents/reward/reward_mean",
            "collection/agents/reward/episode_reward_max",
            "collection/agents/reward/episode_reward_mean",
            "collection/agents/reward/episode_reward_max",
            "collection/agents/social_influenced_reward/social_influenced_reward_max",
            "collection/agents/social_influenced_reward/social_influenced_reward_mean",
            "collection/agents/social_influenced_reward/social_influenced_reward_min",
            "collection/agents/taxed_return/taxed_return_mean",
            "collection/agents/taxed_return/taxed_return_min",
            "collection/agents/taxed_return/taxed_return_max",
            "collection/agents/taxed_reward/taxed_reward_max",
            "collection/agents/taxed_reward/taxed_reward_min",
            "collection/agents/taxed_reward/taxed_reward_mean",
            "collection/reward/episode_reward_mean",
            "collection/reward/episode_reward_min",
            "collection/reward/episode_reward_max",
            "eval/agents/reward/episode_reward_max",
            "eval/agents/reward/episode_reward_min",
            "eval/agents/reward/episode_reward_mean",
            "eval/reward/episode_reward_min",
            "eval/reward/episode_reward_max",
            "eval/reward/episode_reward_mean",
        ]
        for header in pertinent_headers:
            data = self.fetch_and_process_sigma_data(header)
            self.visualize_individual_sigma_data(data, header)