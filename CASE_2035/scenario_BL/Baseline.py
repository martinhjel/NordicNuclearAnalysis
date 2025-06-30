from functions.global_functions import *

# Define global variables
CASE_YEAR = 2035
SCENARIO = 'BL'
VERSION = 'v41'

SIM_YEAR_START = 1991
SIM_YEAR_END = 2020
DATE_START = pd.Timestamp(f'{SIM_YEAR_START}-01-01 00:00:00', tz='UTC')
DATE_END = pd.Timestamp(f'{SIM_YEAR_END}-12-31 23:00:00', tz='UTC')

loss_method = 0
solver = 'glpk'


# GET BASE DIRECTORY
try:
    # FOR SCRIPTS
    BASE_DIR = pathlib.Path(__file__).parent
except NameError:
    # FOR NOTEBOOKS AND INTERACTIVE SHELLS
    BASE_DIR = pathlib.Path().cwd()
    BASE_DIR = BASE_DIR / f'CASE_{CASE_YEAR}' / f'scenario_{SCENARIO}'

# SQL PATH
SQL_FILE = BASE_DIR / f"powergama_{SCENARIO}_{VERSION}_{SIM_YEAR_START}_{SIM_YEAR_END}.sqlite"

# FILE PATHS
DATA_PATH = BASE_DIR / 'data'
OUTPUT_PATH = BASE_DIR / 'results'
OUTPUT_PATH_PLOTS = BASE_DIR / 'results' / 'plots'


# %%  === Configure Grid and Run Simulation ===
data, time_max_min = setup_grid(VERSION, DATE_START, DATE_END, DATA_PATH, SCENARIO)
res = solve_lp(data, SQL_FILE, loss_method, replace=True, solver=solver)
