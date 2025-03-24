# Imports
from powergama.database import Database  # Import Database-Class specifically
from functions.global_functions import *
from scripts.case_doc import *


# Define global variables
YEAR_SCENARIO = 2025
case = 'Equivalent'
version = 'v1'

YEAR_START = 2020
YEAR_END = 2020

# SQL_FILE = "powergama_2025_30y_v1.sqlite"
# DATE_START = f"{YEAR_START}-01-01"
DATE_START = pd.Timestamp(f'{YEAR_START}-01-01 00:00:00', tz='UTC')

# DATE_END = f"{YEAR_END}-01-02"
DATE_END = pd.Timestamp(f'{YEAR_END}-02-01 23:00:00', tz='UTC')


loss_method = 0
new_scenario = False
save_scenario = False


# Get the base directory
try:
    # For scripts
    BASE_DIR = pathlib.Path(__file__).parent
except NameError:
    # For notebooks or interactive shells
    BASE_DIR = pathlib.Path().cwd()
    BASE_DIR = BASE_DIR / f'case_{case}'


SQL_FILE = BASE_DIR / f"powergama_{case}_{version}_{YEAR_START}_{YEAR_END}.sqlite"


# File paths
DATA_PATH = BASE_DIR / 'data'
OUTPUT_PATH = BASE_DIR / 'results'
OUTPUT_PATH_PLOTS = BASE_DIR / 'results' / 'plots'




data, time_max_min = setup_grid(YEAR_SCENARIO, version, DATE_START, DATE_END, DATA_PATH, new_scenario, save_scenario, case)
res = solve_lp(data, SQL_FILE, loss_method, replace=True, nuclear_availability=None, week_MSO=None)


# %%

output_path = OUTPUT_PATH / f'prices_and_branch_utilization_map_{version}.html'
create_price_and_utilization_map(data, res, time_max_min=time_max_min, output_path=output_path, dc=None)
