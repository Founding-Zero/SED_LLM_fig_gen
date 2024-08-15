import os
import wandb
import dataclasses
from dotenv import load_dotenv
from typing import List
from argparse import Namespace
from dataclasses import field, dataclass
from eztils.argparser import HfArgumentParser, update_dataclass_defaults
from rich import print
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


load_dotenv()


def setup_experiment():
    """
    Sets up the experiment by creating a run directory and a log directory, and creating a symlink from the repo directory to the run directory.
    """
    print("Setting up experiment...")
    
    """SETUP CONFIG"""
    parser = HfArgumentParser(Config)
    parser.add_argument("-c", "--config", type=str)

    conf: Config
    extras: Namespace
    conf, extras = parser.parse_args_into_dataclasses()

    if extras.config is not None:  # parse config file
        (original_conf,) = parser.parse_json_file(extras.config)
        for field_ in dataclasses.fields(original_conf):
            val = getattr(original_conf, field_.name)
            if isinstance(val, list):
                setattr(
                    field_, "default_factory", lambda x=val: x
                )
                setattr(original_conf, field_.name, field_)
        parser = HfArgumentParser(update_dataclass_defaults(Config, original_conf))
        parser.add_argument("-c", "--config", type=str)
        conf, extras = parser.parse_args_into_dataclasses()

    return conf


@dataclass
class Config:
    gridsearch: bool = False # Toggle to true to pull the configs from wandb (only needs to be done once)
    pull_configs: bool = False # Toggle to true to pull the configs from wandb (only needs to be done once)
    pulling_data: bool = False # Toggle to true to pull data from wandb
    plotting: bool = False # Toggle to true to plot data
    clip_axis: bool = False # Toggle to plot with the x_axis clipped to so all line end at same point
    entity: str = ""
    project: List[str] = field(default_factory=lambda: ["llm-tests"]) # Tags to filter runs by
    tags: List[str] = field(default_factory=lambda: None) # Tags to filter runs by
    run_ids: List[str] = field(default_factory=lambda: None) # List of run ids to pull data from or None to pull all runs from the project
    methods: List[str] = field(default_factory=lambda: ["LLM", "Random", "Dual-RL", "AID"])
    axis: str = "_step"
    metric: str = ""
    metrics_by_method: List[str] = field(default_factory=lambda: [""])
    custom_palette: List[str] = field(default_factory=lambda: ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"])


if __name__ == "__main__":
    # Multiply episode counters by num_parallel_games when plotting
    
    print(f"[bold green]Welcome to SED LLM Figgen")

    cfg = setup_experiment()
    print(cfg)

    from figgen.sed_llm import SED_DataAnalyzer, process_dataframe
    if cfg.pulling_data:
        for proj in cfg.project:
            sed_llm_analyzer = SED_DataAnalyzer(cfg.entity, proj)
            sed_llm_analyzer.pull_and_store_wandb_data(cfg)
        
    if cfg.plotting:  
        from figgen.visualize import plot_principal_principal, plot_principal_principal_hparams
        df_dict, group_names = process_dataframe(cfg)
        if cfg.gridsearch:
            plot_principal_principal_hparams(df_dict, groups_list=group_names, cfg=cfg)
        else:
            plot_principal_principal(df_dict, groups_list=group_names, cfg=cfg)

