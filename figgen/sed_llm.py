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
import dataclasses
from datetime import datetime, timedelta
from dotenv import load_dotenv
from typing import List, Optional
import copy
import json
import time
import pickle
from argparse import Namespace
from collections import defaultdict
from dataclasses import dataclass, field, make_dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from eztils import abspath, datestr, setup_path
from eztils.argparser import HfArgumentParser, update_dataclass_defaults
from rich import print

from figgen.data_analyzer import DataAnalyzer
from figgen.visualize import plot_principal_principal
from figgen import Config

class SED_DataAnalyzer(DataAnalyzer):
    # def __init__(self, wandb_entity, wandb_project):
    #     super().__init__(wandb_entity, wandb_project)
    #     self.csv_paths = []
        
    def get_runs_by_tag(self, tag):
        # Query runs from the specified entity and project
        runs = self.api.runs(f"{self.wandb_entity}/{self.wandb_project}")

        # Collect run information with the specified tag
        run_data = []
        for run in runs:
            if tag in run.tags:  
                # run_info = {
                #     "id": run.id,
                #     "name": run.name,
                #     "state": run.state,
                #     "created_at": run.created_at,
                #     "summary": run.summary,
                #     "config": run.json_config,
                # }
                # run_data.append(run_info)   
                run_data.append(run)   
                # print(f"Run ID: {run.id}, Name: {run.name}, State: {run.state}")
        self.runs = run_data
        return self.runs

    def pull_run_data(self, cfg: Config):
        self.set_runs(run_ids=cfg.run_ids)
        self.set_histories()

    def pull_and_store_wandb_data(self, cfg: Config):
        """
        function to pull all wandb data
        """
        self.pull_run_data(cfg)
        for run in self.runs:
            os.mkdir(f'figgen/pulled_data/{run.name}')
            with open(f'figgen/pulled_data/{run.name}/config.json', 'w') as json_file:
                json.dump(run.config, json_file, indent=4)
                
            history_df = self.histories[run.id]
            # relevant_headers = []
            # prefixes = cfg.prefixes
            # for column in history_df.columns:
            #     if column.startswith(tuple(prefixes)):
            #         relevant_headers.append(column)
            csv_file_path = f'figgen/pulled_data/{run.name}/csv_dataframe.csv'
            # history_df[relevant_headers].to_csv(csv_file_path, index=False)
            history_df.to_csv(csv_file_path, index=True)
            # self.csv_paths.append(csv_file_path)
   
    
def process_dataframe(cfg: Config):
    # pulling the data and configs per run
    def pull_config_jsons():
        config_jsons = []
        dfs = []
        for run in os.listdir("figgen/pulled_data"):
            item_path = os.path.join("figgen/pulled_data", run)
            if os.path.isdir(item_path):
                
                config_file_path = os.path.join(item_path, 'config.json')
                if os.path.exists(config_file_path):
                    # Read and process the config file
                    with open(config_file_path, 'r') as config_file:
                        config_json = json.load(config_file)
                        config_jsons.append(config_json)
                        
                csv_file_path = os.path.join(item_path, 'csv_dataframe.csv')
                if os.path.exists(csv_file_path):
                    df = pd.read_csv(csv_file_path)
                    df.dropna(how='all', inplace=True)
                    dfs.append(df)
                        
        return config_jsons, dfs
                
    # group the dataframes by the config ignoring seed
    def group_df_objects(json_list, df_list):
        def get_cfg_key(obj):
            # Create a new dictionary without the 'seed' field
            return json.dumps({k: v for k, v in obj.items() if k != 'seed'}, sort_keys=True)

        groups = defaultdict(list)
        for i in range(len(json_list)):
            conf_key = get_cfg_key(json_list[i])
            # append the associated dataframe instead of the config
            groups[conf_key].append(df_list[i])
    
        return groups
    
    # pull the data per group
    conf_json_list, df_list = pull_config_jsons()
    grouped = group_df_objects(conf_json_list, df_list)
    
    df_dict = {}
    group_names = []
    for conf, df_list in grouped.items():
        conf = json.loads(conf)
        if conf['principal'] == 'LLM':
            group_name = f"{conf['env_name']}_{conf['principal']}_{conf['llm_prompt_style']}"
        else:
            group_name = f"{conf['env_name']}_{conf['principal']}"
        
        combined_df = pd.concat(df_list)
        df_dict[group_name] = combined_df
        group_names.append(group_name)
        
    return df_dict, group_names
    plot_principal_principal(df_dict, x_col=cfg.axis, y_col=cfg.metric, groups=group_names)
    
    # with open(f"dataframe.pkl", "wb") as outfile: 
    #     pickle.dump(df, outfile)
    
    