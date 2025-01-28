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
print(f"System cost {sum(getSystemCostFromDB(data=data, db=database, timeMaxMin=time_max_min).values()):.2f} EUR, or {sum(getSystemCostFromDB(data=data, db=database, timeMaxMin=time_max_min).values())/1e9:.2f} Billion EUR")

print(f"Mean area price {sum(getAreaPricesAverageFromDB(data=data, db=database, areas=None, timeMaxMin=time_max_min).values()) / len(getAreaPricesAverageFromDB(data=data, db=database, areas=None, timeMaxMin=time_max_min)):.2f} EUR/MWh")





# %%

storfilling = pd.DataFrame()
areas = ["NO"]          # When plotting multiple years in one year, recommend to only use one area
relative=True           # Relative storage filling, True gives percentage
interval=1              # Month interval for x-axis if plot_by_year is False
plot_by_year = True    # True: Split plot by year, or False: Plot all years in one plot
duration_curve = False  # True: Plot duration curve, or False: Plot storage filling over time
save_plot_SF = False    # True: Save plot as pdf

for area in areas:
    storfilling[area] = getStorageFillingInAreasFromDB(data=data, db=database, areas=[area], generator_type="hydro", relative_storage=relative, timeMaxMin=time_max_min)
    if relative:
        storfilling[area] = storfilling[area] * 100
storfilling.index = pd.date_range(DATE_START, periods=time_max_min[-1], freq='h')
storfilling['year'] = storfilling.index.year    # Add year column to DataFrame
title_storage_filling = f'Reservoir Filling in {areas} for period {DATE_START[0:4]}-{DATE_END[0:4]}'
plot_storage_filling_area(storfilling=storfilling,
                          DATE_START=DATE_START,
                          DATE_END=DATE_END,
                          areas=areas,
                          interval=interval,
                          title=title_storage_filling,
                          OUTPUT_PATH_PLOTS=OUTPUT_PATH_PLOTS,
                          relative=relative,
                          plot_by_year=plot_by_year,
                          save_plot=save_plot_SF,
                          duration_curve=duration_curve,
                          tex_font=False)