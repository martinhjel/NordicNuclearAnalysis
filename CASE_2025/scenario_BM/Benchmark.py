# Imports
from powergama.database import Database  # Import Database-Class specifically
from functions.global_functions import *
from scripts.case_doc import *


# Define global variables
CASE_YEAR = 2025
SCENARIO = 'BM'
VERSION = 'nuclear_SE4_1_v1'

SIM_YEAR_START = 1991
SIM_YEAR_END = 2020
DATE_START = pd.Timestamp(f'{SIM_YEAR_START}-01-01 00:00:00', tz='UTC')
DATE_END = pd.Timestamp(f'{SIM_YEAR_END}-12-31 23:00:00', tz='UTC')

loss_method = 0

# Get the base directory
try:
    # For scripts
    BASE_DIR = pathlib.Path(__file__).parent
except NameError:
    # For notebooks or interactive shells
    BASE_DIR = pathlib.Path().cwd()
    BASE_DIR = BASE_DIR / f'case_{CASE_YEAR}' / f'scenario_{SCENARIO}'

SQL_FILE = BASE_DIR / f"powergama_{SCENARIO}_{VERSION}_{SIM_YEAR_START}_{SIM_YEAR_END}.sqlite"

# File paths
DATA_PATH = BASE_DIR / 'data'
OUTPUT_PATH = BASE_DIR / 'results'
OUTPUT_PATH_PLOTS = BASE_DIR / 'results' / 'plots'

# TODO: week_MSO bør legges inn i generatorfilen og leses inn som en del av GridData.
week_MSO = {'FI_10':16,
            'FI_12':36,
            'SE3_3':20,
            'SE3_6':24,
            'GB':32,
            'NL':28
            }


# %%
# Configure grid and run simulation
# create_case_doc('BM') # Create case documentation
data, time_max_min = setup_grid(VERSION, DATE_START, DATE_END, DATA_PATH, SCENARIO)
res = solve_lp(data, SQL_FILE, loss_method, replace=True, nuclear_availability=0.7, week_MSO=week_MSO)

