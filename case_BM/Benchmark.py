# Imports
from powergama.database import Database  # Import Database-Class specifically
from functions.global_functions import *
from scripts.case_doc import *


# Define global variables
YEAR_SCENARIO = 2025
case = 'BM'
version = 'v80'

YEAR_START = 1991
YEAR_END = 1993

# SQL_FILE = "powergama_2025_30y_v1.sqlite"
# DATE_START = f"{YEAR_START}-01-01"
DATE_START = pd.Timestamp(f'{YEAR_START}-01-01 00:00:00', tz='UTC')

# DATE_END = f"{YEAR_END}-01-02"
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


SQL_FILE = BASE_DIR / f"powergama_{case}_{version}_{YEAR_START}_{YEAR_END}.sqlite"


# File paths
DATA_PATH = BASE_DIR / 'data'
OUTPUT_PATH = BASE_DIR / 'results'
OUTPUT_PATH_PLOTS = BASE_DIR / 'results' / 'plots'


# week_start = np.random.permutation([16, 20, 24, 28, 32, 36])  # Middle of Apr, May, Jun, Jul, Aug, Sep
# Randomised week sequence, allocated to NNP in indexed order,
# meaning first index in week_start is allocated to first indexed NNP.
# This means finland gets first go.
# week_start = [16, 36, 20, 24, 32, 28]
# week Maintenance Start Order
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
data, time_max_min = setup_grid(YEAR_SCENARIO, version, DATE_START, DATE_END, DATA_PATH, new_scenario, save_scenario, case)
res = solve_lp(data, SQL_FILE, loss_method, replace=True, nuclear_availability=0.7, week_MSO=week_MSO)

# res.getEnergyBalanceInArea(area='NO', spillageGen='wind_on')

# %% Print results
# print(f"System cost {sum(res.getSystemCost().values()):.2f} EUR, or {sum(res.getSystemCost().values())/1e9:.2f} Billion EUR")
# print(f"Mean area price {sum(res.getAreaPricesAverage().values()) / len(res.getAreaPricesAverage()):.2f} EUR/MWh")


# %%

output_path = OUTPUT_PATH / f'prices_and_branch_utilization_map_{version}.html'
create_price_and_utilization_map(data, res, time_max_min=time_max_min, output_path=output_path, dc=1)



