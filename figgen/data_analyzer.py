import os
import tempfile
from collections import defaultdict
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import wandb
from tqdm import tqdm
from dotenv import load_dotenv


class DataAnalyzer:
    def __init__(
        self,
        wandb_entity,
        wandb_projects,
        color_scheme="viridis",
        export_to_wandb=False,
    ):
        load_dotenv()
        self.wandb_entity = wandb_entity
        self.wandb_project = wandb_projects
        self.api_key = os.getenv("WANDB_API_KEY")
        self.api = wandb.Api(timeout=90)
        self.histories = {}
        self.color_scheme = color_scheme
        self.export_to_wandb = export_to_wandb
        if self.export_to_wandb and self.api_key:
            wandb.init(project=self.wandb_project, entity=self.wandb_entity)

    def set_runs(self, run_ids=None):
        if run_ids is not None:
            self.runs = [
                self.api.run(f"{self.wandb_entity}/{self.wandb_project}/{run_id}")
                for run_id in run_ids
            ]
        else:
            self.runs = self.api.runs(f"{self.wandb_entity}/{self.wandb_project}")
        return self.runs

    def send_to_wandb(self, fig, title):
        if self.export_to_wandb:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                fig_path = tmpfile.name
                fig.savefig(fig_path)
                wandb.log({title: wandb.Image(fig_path)})
                os.unlink(fig_path)

    def set_histories(self):
        for run in tqdm(self.runs, desc="Processing Runs", unit="run"):
            # if int(run.name.split("-")[-1]) > 159:
            self.histories[run.id] = run.scan_history()
            # with open(f"dataframe_{run.name}.pkl", "wb") as outfile: 
            #     pickle.dump(self.histories[run.id], outfile)      
        return self.histories

    def visualize_lineplot_groupby(
        self,
        title: str,
        x_key: str,
        y_key: str,
        group_key: str,
        data: pd.DataFrame,
        x_label: str = None,
        y_label: str = None,
        x_ticks_by_data: bool = False,
    ):
        groups = data[group_key].unique()
        palette = sns.color_palette(self.color_scheme, n_colors=len(groups))
        color_dict = {skill_level: color for skill_level, color in zip(groups, palette)}
        with sns.axes_style("darkgrid"):
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.lineplot(
                data=data,
                x=x_key,
                y=y_key,
                hue=group_key,
                dashes=False,
                palette=color_dict,
                err_style="band",
            )

            plt.title(title, fontsize="large")
            plt.xlabel(x_label or x_key.capitalize(), fontsize="large")
            plt.ylabel(y_label or y_key.capitalize(), fontsize="large")

            # plt.xlim(left=0, right=self.min_length)
            if x_ticks_by_data:
                plt.xticks(data[x_key].unique())

            plt.ylim(bottom=0)
            plt.grid(True)
            if self.export_to_wandb:
                self.send_to_wandb(fig, title)
            else:
                plt.savefig(f"{title}.png")