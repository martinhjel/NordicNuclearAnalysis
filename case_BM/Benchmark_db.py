from powergama.database import Database  # Import Database-Class specifically
from functions.global_functions import *  # Functions like 'read_grid_data', 'solve_lp' m.m.
from functions.database_functions import  * # Functions like 'getSystemCostFromDB' m.m.


YEAR_SCENARIO = 2025
YEAR_START = 1991
YEAR_END = 2020
case = 'BM'
version = 'v2'
SQL_FILE = f"powergama_{case}_{version}.sqlite"
# DATE_START = f"{YEAR_START}-01-01"
DATE_START = pd.Timestamp(f'{YEAR_START}-01-01 00:00:00', tz='UTC')
# DATE_END = f"{YEAR_END}-12-31"
DATE_END = pd.Timestamp(f'{YEAR_END}-12-31 23:00:00', tz='UTC')
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

# File paths
DATA_PATH = BASE_DIR / 'data'
OUTPUT_PATH = BASE_DIR / 'results'
OUTPUT_PATH_PLOTS = BASE_DIR / 'results' / 'plots'


data, time_max_min = setup_grid(YEAR_SCENARIO, version, DATE_START, DATE_END, DATA_PATH, new_scenario, save_scenario)
database = Database(SQL_FILE)

x = 1
# %%
print(f"System cost {sum(getSystemCostFromDB(data, database, time_max_min).values()):.2f} EUR, or {sum(getSystemCostFromDB(data, database, time_max_min).values())/1e9:.2f} Billion EUR")
