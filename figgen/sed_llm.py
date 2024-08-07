import os
import random
import csv
import tempfile
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from datetime import datetime, timedelta
from figgen import DataAnalyzer
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
        self.api = wandb.Api()
        self.histories = {}
        self.color_scheme = color_scheme
        self.export_to_wandb = export_to_wandb
        if self.export_to_wandb and self.api_key:
            wandb.init(project=self.wandb_project, entity=self.wandb_entity)

    def get_runs(self, run_ids=None):
        if run_ids is not None:
            self.runs = [
                self.api.run(f"{self.wandb_entity}/{self.wandb_project}/{run_id}")
                for run_id in run_ids
            ]
        else:
            self.runs = self.api.runs(f"{self.wandb_entity}/{self.wandb_project}")
        return self.runs

    def get_runs_by_tag(self, tag):
        # Query runs from the specified entity and project
        runs = self.api.runs(f"{entity}/{project}")

        # Collect run information with the specified tag
        run_data = []
        for run in runs:
            if tag in run.tags:
                # run_info = {
                #     "id": run.id,
                #     "name": run.name,
                #     "state": run.state,
                #     "config": run.config,
                #     "summary": run.summary,
                #     "created_at": run.created_at,
                # }
                # run_data.append(run_info)   
                run_data.append(run)   
        self.runs = run_data
        return self.runs

    def send_to_wandb(self, fig, title):
        if self.export_to_wandb:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                fig_path = tmpfile.name
                fig.savefig(fig_path)
                wandb.log({title: wandb.Image(fig_path)})
                os.unlink(fig_path)

    def get_histories(self):
        for run in self.runs:
            self.histories[run.id] = run.history()

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
                

if __name__ == "__main__":
    wandb.login(os.getenv("WANDB_API_KEY"))
    entity = "lad"
    project = "llm-tests"
    tag = "prompt_difficulty"
    sed_llm_analyzer = DataAnalyzer(entity, project)
    
    runs_with_tag = sed_llm_analyzer.get_runs_by_tag(tag)
    for run in runs_with_tag:
        print(f"Run ID: {run.id}, Name: {run.name}, State: {run.state}")
