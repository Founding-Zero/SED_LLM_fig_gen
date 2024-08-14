# AI Principal Figgen

## Installation

```
make install
conda activate py311
source .venv/bin/activate.fish
pip install -r requirements.txt
pip install tqdm pandas python-dotenv rich seaborn matplotlib wandb

mkdir output_plots
```

## Run
```
- Adjust the config for which wandb project you want to pull from. 

- Note, this is a 2 step process (pull from wandb and plot). 

- You can run both steps by setting the first 3 args to true. If you just want to plot after pulling the data set the first 2 args to false and only set the 3rd arg to true.

- Run with debugger. launch.json is already configured.
```
