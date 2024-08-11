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
    pulling_data: bool = False
    plotting: bool = False
    entity: str = ""
    project: str = ""
    tag: str = ""
    run_ids: List[str] = field(default_factory=lambda: [""])
    env: str = ""
    method: str = ""
    axis: str = "_step"
    metric: str = ""
    prefixes: List[str] = field(default_factory=lambda: [""])


if __name__ == "__main__":
    # Multiply episode counters by num_parallel_games when plotting
    
    print(f"[bold green]Welcome to SED LLM Figgen")

    cfg = setup_experiment()
    print(cfg)

    from figgen.sed_llm import SED_DataAnalyzer, process_dataframe
    if cfg.pulling_data:
        sed_llm_analyzer = SED_DataAnalyzer(cfg.entity, cfg.project)
        sed_llm_analyzer.pull_and_store_wandb_data(cfg)
        
    if cfg.plotting:  
        from figgen.visualize import plot_principal_principal
        df_dict, group_names = process_dataframe(cfg)
        plot_principal_principal(df_dict, x_col=cfg.axis, y_col=cfg.metric, groups=group_names, title='Principal Return vs Episode')

