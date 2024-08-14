import os
import random
import csv
import tempfile
from collections import defaultdict
import pandas as pd
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
from rich import print
from tqdm import tqdm
from figgen.data_analyzer import DataAnalyzer
from figgen import Config

class SED_DataAnalyzer(DataAnalyzer):        
    def get_runs_filtered_by_tag(self, cfg: Config):
        # Query runs from the specified entity and project
        # runs = self.api.runs(f"{self.wandb_entity}/{self.wandb_project}")
        runs = self.set_runs(run_ids=cfg.run_ids)
        # Collect run information with the specified tag
        run_data = []
        for run in runs:
            if cfg.tag in run.tags:    
                run_data.append(run)   
        self.runs = run_data
        return self.runs

    def pull_and_store_wandb_data(self, cfg: Config):
        """
        function to pull all wandb run data and config files for the runs
        """
        # self.pull_run_data(cfg)
        # self.set_runs(run_ids=cfg.run_ids)
        # pull the runs:
        self.get_runs_filtered_by_tag(cfg=cfg)
        for run in tqdm(self.runs):
            # create run file (named by run.name) and store the config file
            if cfg.pull_configs:
                if os.path.exists(f'figgen/pulled_data/{run.name}') and os.path.isdir(f'figgen/pulled_data/{run.name}'):
                    continue
                os.mkdir(f'figgen/pulled_data/{run.name}')
                with open(f'figgen/pulled_data/{run.name}/config.json', 'w') as json_file:
                    json.dump(run.config, json_file, indent=4)
            
            # pull the principal return plot data and store in CSVs
            if 'LLM' in run.tags or 'random' in run.tags:
                keys = [cfg.metrics_by_method[0], 'principal_final/principal_step']
            elif 'aid' in run.tags or 'dual_rl' in run.tags: 
                keys = [cfg.metrics_by_method[1], 'principal_final/principal_step']

            axis_multiplier = 2 if 'aid' in run.tags else 20
            all_history = []
            for row in run.scan_history(keys=keys):
                all_history.append(row)
            history_df = pd.DataFrame(all_history)  
            if cfg.axis == 'combined_val_train/episode':
                history_df['principal_final/principal_step'] *= axis_multiplier
            csv_file_path = f'figgen/pulled_data/{run.name}/csv_dataframe.csv'
            history_df.to_csv(csv_file_path, index=False)
   
    
def process_dataframe(cfg: Config):
    """
    pulling the data and configs per run and transform them into the correct format for plotting
    """
    
    # pull the config jsons and dataframes
    def pull_config_jsons():
        config_jsons = []
        dfs = []
        for run in tqdm(os.listdir("figgen/pulled_data")):
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
            if obj['principal'] == 'Random':
                return json.dumps({k: v for k, v in obj.items()}, sort_keys=True)
            else:
                # Create a new dictionary without the 'seed' field for every other method besides Random
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
        conf = json.loads(conf) # json string to python dict
        if conf['principal'] == 'LLM':
            # group_name = f"{conf['env_name']}_{conf['principal']}_ps_{conf['temperature']}"
            group_name = f"{conf['env_name']}_{conf['principal']}_ps_{conf['llm_prompt_style']}"
        elif conf['principal'] == 'Random':
            # group_name = f"{conf['env_name']}_{conf['principal']}_{conf['seed']}"
            group_name = f"{conf['env_name']}_{conf['principal']}"
        else: # Dual-RL and AID
            group_name = f"{conf['env_name']}_{conf['principal']}_{conf['principal_lr']}"
        
        combined_df = pd.concat(df_list) # combine the dataframes to get std
        df_dict[group_name] = combined_df
        group_names.append(group_name)
        
    return df_dict, group_names    
    