from functions.work_functions import *
from functions.global_functions import *  # Functions like 'read_grid_data', 'solve_lp' m.m.
from functions.database_functions import  * # Functions like 'getSystemCostFromDB' m.m.
from zoneinfo import ZoneInfo
from powergama.database import Database  # Import Database-Class specifically
import pandas as pd
import numpy as np


# === General Configurations ===
SIM_YEAR_START = 1991           # Start year for the main simulation  (SQL-file)
SIM_YEAR_END = 1993            # End year for the main simulation  (SQL-file)
CASE_YEAR = 2035
SCENARIO = 'FDT'
VERSION = 'v1_sens'
TIMEZONE = ZoneInfo("UTC")  # Definerer UTC tidssone

DATE_START = pd.Timestamp(f'{SIM_YEAR_START}-01-01 00:00:00', tz='UTC')
DATE_END = pd.Timestamp(f'{SIM_YEAR_END}-12-31 23:00:00', tz='UTC')

loss_method = 0

# Get base directory dynamically
try:
    # For scripts
    BASE_DIR = pathlib.Path(__file__).parent
except NameError:
    # For notebooks or interactive shells
    BASE_DIR = pathlib.Path().cwd()
    BASE_DIR = BASE_DIR / f'case_{CASE_YEAR}' / f'scenario_{SCENARIO}'

# === File Paths ===
SQL_FILE = BASE_DIR / f"powergama_{SCENARIO}_{VERSION}_{SIM_YEAR_START}_{SIM_YEAR_END}.sqlite"
DATA_PATH = BASE_DIR / 'data'
GRID_DATA_PATH = DATA_PATH / 'system'
OUTPUT_PATH = BASE_DIR / 'results'
OUTPUT_PATH_PLOTS = BASE_DIR / 'results' / 'plots'

# === Initialize Database and Grid Data ===
data, time_max_min = setup_grid(VERSION, DATE_START, DATE_END, DATA_PATH, SCENARIO)
database = Database(SQL_FILE)


# %% === Nordic Grid Map ===

# === INITIALIZATIONS ===
START = {"year": 1991, "month": 1, "day": 1, "hour": 0}
END = {"year": 1993, "month": 12, "day": 31, "hour": 23}
nordic_grid_map_fromDB(data, database, time_range = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END),
                       OUTPUT_PATH = OUTPUT_PATH, version = VERSION, START = START, END = END, exchange_rate_NOK_EUR = 11.38)


# %% === ZONAL PRICE MAP ===
zones = ['NO1', 'NO2', 'NO3', 'NO4', 'NO5', 'SE1', 'SE2', 'SE3', 'SE4',
         'DK1', 'DK2', 'FI', 'DE', 'GB', 'NL', 'LT', 'PL', 'EE']
year_range = list(range(SIM_YEAR_START, SIM_YEAR_END + 1))
price_matrix, log = createZonePriceMatrix(data, database, zones, year_range, TIMEZONE, SIM_YEAR_START, SIM_YEAR_END)
# Plot Zonal Price Matrix
plotZonePriceMatrix(price_matrix, save_fig=True, OUTPUT_PATH_PLOTS=OUTPUT_PATH_PLOTS, start=SIM_YEAR_START, end=SIM_YEAR_END, version=VERSION)


# %% STORAGE - PLOT STORAGE FILLING FOR AREAS

# === INITIALIZATIONS ===
START = {"year": 1991, "month": 1, "day": 1, "hour": 0}
END = {"year": 1993, "month": 12, "day": 31, "hour": 23}

# === PLOT CONFIGURATIONS ===
plot_config = {
    'areas': ['NO'],            # When plotting multiple years in one year, recommend to only use one area
    'relative': True,           # Relative storage filling, True gives percentage
    "plot_by_year": True,       # True: One curve for each year in same plot, or False:all years collected in one plot over the whole simulation period
    "duration_curve": False,    # True: Plot duration curve, or False: Plot storage filling over time
    "save_fig": False,           # True: Save plot as pdf
    "interval": 1,              # Number of months on x-axis. 1 = Step is one month, 12 = Step is 12 months
    'empty_threshold': 1e-6,    # If relative (True), empty_threshold is in percentage, if not, it is in MWh
    'include_legend': False,     # Include legend in the plot
    'fig_size': (10, 6),        # Figure size in inches
    'tex_font': False,          # Keep false unless tex packages are installed.
                                # Kan hende må kjøres et par ganger for å få det til å funke med texfont.
}

# === COMPUTE TIMERANGE AND PLOT FLOW ===
time_SF = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
plot_SF_Areas_FromDB(data, database, time_SF, OUTPUT_PATH_PLOTS, DATE_START, plot_config)

# %% STORAGE - PLOT STORAGE FILLING ZONES
# Todo: Trengs det fortsatt litt jobb med scaleringen av selve plottet, men det er ikke krise enda.
# Todo: Må OGSÅ ha mulighet til å plotte storage filling ned på node nivå.

# === INITIALIZATIONS ===
START = {"year": 1991, "month": 1, "day": 1, "hour": 0}
END = {"year": 2020, "month": 12, "day": 31, "hour": 23}

# === PLOT CONFIGURATIONS ===
plot_config = {
    'zones': ['NO5'],               # When plotting multiple years in one year, recommend to only use one zone
    'relative': True,               # Relative storage filling, True gives percentage
    "plot_by_year": 3,              # (1) Each year in individual plot, (2) Entire Timeline, (3) Each year show over 1 year timeline.
    "duration_curve": False,        # True: Plot duration curve, or False: Plot storage filling over time
    "save_fig": False,              # True: Save plot as pdf
    "interval": 1,                  # Number of months on x-axis. 1 = Step is one month, 12 = Step is 12 months
    'empty_threshold': 1e-6,        # If relative (True), empty_threshold is in percentage, if not, it is in MWh
    'include_legend': False,        # Include legend in the plot
    'fig_size': (10, 6),            # Figure size in inches
    'tex_font': False,              # Keep false unless tex packages are installed
}

# If you want to go in and change title, follow the function from here to its source location and change it there.
# Remember that you then have to reset the console run
# === COMPUTE TIMERANGE AND PLOT FLOW ===
time_SF = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
plot_SF_Zones_FromDB(data, database, time_SF, OUTPUT_PATH_PLOTS, DATE_START, plot_config)


# %% Yearly national-level electricity production by type and consumption
"""
Retrieves and aggregates electricity production and demand data at the national level.

Based on idealyears over the 30-year climate periode so that each year has same number of hours so it can be comparable to each other.
BASERT PÅ TIDSSTEG OG IKKE DATO! 
"""

# === INITIALIZATIONS ===
country = "SE"  # Country code

n_ideal_years = 1
n_timesteps = int(8760 * n_ideal_years) # Ved full 30-års simuleringsperiode

df_gen, df_prices, total_production, df_gen_per_year = get_production_by_type_ideal_timestep(
    data=data,
    db=database,
    area_OP=country,
    n_timesteps=n_timesteps
)

# %% Sensitivities Nuclear production

# === INITIALIZATIONS ===
START = {"year": 1991, "month": 1, "day": 1, "hour": 0}
END = {"year": 1993, "month": 12, "day": 31, "hour": 23}

# Sum sensitivity [€] over all time steps
df_NuclearSens_raw = database.getResultNuclearSens(get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END))
df_sens_nuclear_gen = df_NuclearSens_raw.sum().reset_index()
df_sens_nuclear_gen.columns = ["generator_idx", "sensitivity [€]"]
df_sens_nuclear_gen["node"] = df_sens_nuclear_gen["generator_idx"].apply(lambda i: data.generator["node"][i])
df_sens_nuclear_gen = df_sens_nuclear_gen[["generator_idx", "node", "sensitivity [€]"]]

# Sensitivity avg. [€/h]
n_timesteps = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)[1]-get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)[0]
df_sens_nuclear_gen["sensitivity avg. [€/h]"] = df_sens_nuclear_gen["sensitivity [€]"] / n_timesteps

# Price avg. [€/MWh]
node_ids = list(data.node["id"])
avg_prices_all = database.getResultNodalPricesMean(get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END))
if len(avg_prices_all) != len(node_ids):
    print(f"⚠️ Number of prices ({len(avg_prices_all)}) does not match number of nodes ({len(node_ids)}). Mapping only the first {min(len(node_ids), len(avg_prices_all))}.")
min_len = min(len(node_ids), len(avg_prices_all))
avg_prices = dict(zip(node_ids[:min_len], avg_prices_all[:min_len]))
df_sens_nuclear_gen["nodal price avg. [€/MWh]"] = df_sens_nuclear_gen["node"].map(avg_prices)

# Nodal price avg. sensitivity diff [€/MWh]
df_sens_nuclear_gen["sensitivity diff [€/MWh]"] = df_sens_nuclear_gen["nodal price avg. [€/MWh]"] + df_sens_nuclear_gen["sensitivity avg. [€/h]"]

# Sort the dataframe by sensitivity
df_sens_nuclear_gen = df_sens_nuclear_gen.sort_values("sensitivity [€]", ascending=True)


# %% GET FLOW ON CHOSEN BRANCHES

# === INITIALIZATIONS ===
START = {"year": 1991, "month": 1, "day": 1, "hour": 0}
END = {"year": 1993, "month": 12, "day": 12, "hour": 23}

# === PLOT CONFIGURATIONS ===

plot_config = {
    "plot_by_year": False,      # Each year in individual plot or all years collected in one plot
    "duration_curve": True,     # True: Plot duration curve, or False: Plot storage filling over time
    "duration_relative": False, # Hours(False) or Percentage(True)
    "save_fig": False,          # True: Save plot as pdf
    "interval": 1,              # Number of months on x-axis. 1 = Step is one month, 12 = Step is 12 months
    "check": False,
    "tex_font": False
}

# === CHOOSE BRANCHES TO CHECK ===
SELECTED_BRANCHES  = [['NL','NO2_4'],['NO2_4','DE'], ['NO2_1','GB'], ['DK1_1','NO2_5']] # See branch CSV files for correct connections

# === COMPUTE TIMERANGE AND PLOT FLOW ===
time_Lines = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
plot_Flow_fromDB(data, database, DATE_START, time_Lines, OUTPUT_PATH_PLOTS, plot_config, SELECTED_BRANCHES)
