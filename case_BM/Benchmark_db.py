from functions.more_functions import *
from functions.global_functions import *  # Functions like 'read_grid_data', 'solve_lp' m.m.
from functions.database_functions import  * # Functions like 'getSystemCostFromDB' m.m.
from zoneinfo import ZoneInfo

#midlertidig:

from powergama.database import Database  # Import Database-Class specifically
from datetime import datetime, timedelta
from powergama.GridData import GridData
import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.chart import LineChart, Reference
import os
from openpyxl import Workbook
from openpyxl.chart import BarChart, LineChart, Reference



# === General Configurations ===
YEAR_SCENARIO = 2025
YEAR_START = 1991           # Start year for the main simulation  (SQL-file)
YEAR_END = 2020             # End year for the main simulation  (SQL-file)
case = 'BM'
version = '52_V16'
TIMEZONE = ZoneInfo("UTC")  # Definerer UTC tidssone


DATE_START = pd.Timestamp(f'{YEAR_START}-01-01 00:00:00', tz='UTC')
DATE_END = pd.Timestamp(f'{YEAR_END}-12-31 23:00:00', tz='UTC')

loss_method = 0
new_scenario = False
save_scenario = False


# Get base directory dynamically
try:
    # For scripts
    BASE_DIR = pathlib.Path(__file__).parent
except NameError:
    # For notebooks or interactive shells
    BASE_DIR = pathlib.Path().cwd()
    BASE_DIR = BASE_DIR / f'case_{case}'



# === File Paths ===
SQL_FILE = BASE_DIR / f"powergama_{case}_{version}.sqlite"
DATA_PATH = BASE_DIR / 'data'
GRID_DATA_PATH = DATA_PATH / 'system'
OUTPUT_PATH = BASE_DIR / 'results'
OUTPUT_PATH_PLOTS = BASE_DIR / 'results' / 'plots'

# === Initialize Database and Grid Data ===
data, time_max_min = setup_grid(YEAR_SCENARIO, version, DATE_START, DATE_END, DATA_PATH, new_scenario, save_scenario)
database = Database(SQL_FILE)




# %% Collect the system cost and mean area price for the system for a given period

# === INITIALIZATIONS ===
START_YEAR = 2000
END_YEAR = 2000

time_SC = get_time_steps_for_period(START_YEAR, END_YEAR)
time_MP = get_time_steps_for_period(START_YEAR, END_YEAR)
calcSystemCostAndMeanPriceFromDB(data, database, time_max_min, time_SC, time_MP)


# %% Map prices and branch utilization

# === INITIALIZATIONS ===
START_YEAR = 2017
END_YEAR = 2017

time_Map = get_time_steps_for_period(START_YEAR, END_YEAR)

# If one smaller period is wanted, choose time_Map = [start, end] for the given timestep you want
plot_Map(data, database, time_Map, DATE_START, OUTPUT_PATH, version)



# %% GET FLOW ON CHOSEN BRANCHES

# === INITIALIZATIONS ===
START_YEAR = 1991
END_YEAR = 2020

# === PLOT CONFIGURATIONS ===

plot_config = {
    'areas': ['NO'],            # When plotting multiple years in one year, recommend to only use one area
    "plot_by_year": False,      # Each year in individual plot or all years collected in one plot
    "duration_curve": True,     # True: Plot duration curve, or False: Plot storage filling over time
    "duration_relative": False, # Hours(False) or Percentage(True)
    "save_fig": False,          # True: Save plot as pdf
    "interval": 1,              # Number of months on x-axis. 1 = Step is one month, 12 = Step is 12 months
    "check": False,
    "tex_font": False
}

# === CHOOSE BRANCHES TO CHECK ===
SELECTED_BRANCHES  = [['DK1_3','DK1_1'],['FI_3','SE1_2'], ['SE4_2','DE']] # See branch CSV files for correct connections


# === COMPUTE TIMERANGE AND PLOT FLOW ===
time_Lines = get_time_steps_for_period(START_YEAR, END_YEAR)
plot_Flow_fromDB(database, DATE_START, time_Lines, GRID_DATA_PATH, OUTPUT_PATH_PLOTS, plot_config, SELECTED_BRANCHES)


# %% PLOT STORAGE FILLING FOR AREAS

# === INITIALIZATIONS ===
START_YEAR = 1991
END_YEAR = 2020

# === PLOT CONFIGURATIONS ===
plot_config = {
    'areas': ['NO'],            # When plotting multiple years in one year, recommend to only use one area
    'relative': True,           # Relative storage filling, True gives percentage
    "plot_by_year": False,       # True: One curve for each year in same plot, or False:all years collected in one plot over the whole simulation period
    "duration_curve": False,    # True: Plot duration curve, or False: Plot storage filling over time
    "save_fig": False,          # True: Save plot as pdf
    "interval": 1               # Number of months on x-axis. 1 = Step is one month, 12 = Step is 12 months
}

# === COMPUTE TIMERANGE AND PLOT FLOW ===
time_SF = get_time_steps_for_period(START_YEAR, END_YEAR)
plot_SF_Areas_FromDB(data, database, time_SF, OUTPUT_PATH_PLOTS, DATE_START, plot_config)

# %% PLOT STORAGE FILLING ZONES

# Her trengs det fortsatt litt jobb med scaleringen av selve plottet, men det er ikke krise enda.
# Todo: Får ikke alle år i et plot for en gitt zone. Eks. NO4 fra 1991 til 2020. Skule hatt et plot med alle år inni.

# === INITIALIZATIONS ===

START_YEAR = 1991
END_YEAR = 2020


# === PLOT CONFIGURATIONS ===
plot_config = {
    'zones': ['NO4'],                             # When plotting multiple years in one year, recommend to only use one zone

    'relative': True,                               # Relative storage filling, True gives percentage
    "plot_by_year": True,                           # Each year in individual plot or all years collected in one plot
    "duration_curve": False,                         # True: Plot duration curve, or False: Plot storage filling over time
    "save_fig": False,                              # True: Save plot as pdf
    "interval": 1                                   # Number of months on x-axis. 1 = Step is one month, 12 = Step is 12 months
}

# If you want to go in and change title, follow the function from here to its source location and change it there.
# Remember that you then have to reset the console run
# === COMPUTE TIMERANGE AND PLOT FLOW ===
time_SF = get_time_steps_for_period(START_YEAR, END_YEAR)
plot_SF_Zones_FromDB(data, database, time_SF, OUTPUT_PATH_PLOTS, DATE_START, plot_config)



# %% Plot nodal prices Norway in a zone


# === INITIALIZATIONS ===
START_YEAR = 2000
END_YEAR = 2000

# === PLOT CONFIGURATIONS ===
plot_config = {
    'zone': 'NO2',                          # When plotting multiple years in one year, recommend to only use one zone
    'plot_all_nodes': False,                # Plot all nodes in a zone or avg of all nodes
    "plot_by_year": False,                  # Each year in individual plot or all years collected in one plot
    "duration_curve": False,                # True: Plot duration curve, or False: Plot storage filling over time
    "save_fig": False,                      # True: Save plot as pdf
    "interval": 1,                          # Number of months on x-axis. 1 = Step is one month, 12 = Step is 12 months
    "tex_font": False

}

# === COMPUTE TIMERANGE AND PLOT FLOW ===
time_NP = get_time_steps_for_period(START_YEAR, END_YEAR)
calcPlot_NP_FromDB(data, database, time_NP, OUTPUT_PATH_PLOTS, DATE_START, plot_config)


# %% PLOT ZONAL PRICES

# === INITIALIZATIONS ===
START_YEAR = 2000
END_YEAR = 2002

# === PLOT CONFIGURATIONS ===
plot_config = {
    'zones': ['NO2', 'NO3', 'NO4', 'NO5'],                # Zones for plotting
    "plot_by_year": True,                  # Each year in individual plot or all years collected in one plot
    "duration_curve": False,                # True: Plot duration curve, or False: Plot storage filling over time
    "save_fig": False,                      # True: Save plot as pdf
    "interval": 1,                          # Number of months on x-axis. 1 = Step is one month, 12 = Step is 12 months
    "tex_font": False                       # Keep false
}

# === COMPUTE TIMERANGE AND PLOT FLOW ===
time_ZP = get_time_steps_for_period(START_YEAR, END_YEAR)
calcPlot_ZonalPrices_FromDB(data, database, time_ZP, OUTPUT_PATH_PLOTS, DATE_START, plot_config)




# %% Hydro production, reservoir filling, inflow


# === INITIALIZATIONS ===
START_YEAR = 2000
END_YEAR = 2000

# === PLOT CONFIGURATIONS ===
plot_config = {
    'area': 'NO',
    'genType': 'hydro',
    'relative_storage': True,
    "plot_full_timeline": False,
    "box_in_frame": False,
    "save_fig": False,                      # True: Save plot as pdf
    "interval": 1                           # Number of months on x-axis. 1 = Step is one month, 12 = Step is 12 months
}

# === COMPUTE TIMERANGE AND PLOT FLOW ===
time_HRI = get_time_steps_for_period(START_YEAR, END_YEAR)
calcPlot_HRI_FromDB(data, database, time_HRI, OUTPUT_PATH_PLOTS, DATE_START, plot_config)


# %% Plot nodal prices, demand and hydro production


# === INITIALIZATIONS ===
START_YEAR = 2000
END_YEAR = 2000

# === PLOT CONFIGURATIONS ===
plot_config = {
    'area': 'NO',
    'title': 'Avg. Area Price, Demand and Hydro Production in NO',
    'resample': True,
    "plot_full_timeline": True,
    "box_in_frame": False,
    "save_fig": False,                      # True: Save plot as pdf
    "interval": 1                           # Number of months on x-axis. 1 = Step is one month, 12 = Step is 12 months
}

# === COMPUTE TIMERANGE AND PLOT FLOW ===
time_PLP = get_time_steps_for_period(START_YEAR, END_YEAR)
calcPlot_PLP_FromDB(data, database, time_PLP, OUTPUT_PATH_PLOTS, DATE_START, plot_config)


# %% Load, generation by type in AREA

# === INITIALIZATIONS ===
START_YEAR = 2000
END_YEAR = 2000

# === PLOT CONFIGURATIONS ===
plot_config = {
    'area': 'NO',
    'title': 'Production, Consumption and Price in NO',
    "fig_size": (10, 6),
    "plot_full_timeline": True,
    "duration_curve": False,
    "box_in_frame": False,
    "save_fig": False,                      # True: Save plot as pdf
    "interval": 1                           # Number of months on x-axis. 1 = Step is one month, 12 = Step is 12 months
}


# === COMPUTE TIMERANGE AND PLOT FLOW ===
time_LGT = get_time_steps_for_period(START_YEAR, END_YEAR)
df_gen_re, df_prices, tot_prod = calcPlot_LG_FromDB(data, database, time_LGT, OUTPUT_PATH_PLOTS, DATE_START, plot_config)
print(f"Total production in {plot_config['area']}: {tot_prod:.2f} MWh")


# %% Production data for area or zone

""" Initialize data for Production"""
# La area eller zone være None om de ikke skal brukes.

# TODO: DENNE FUNGERER KUN FOR SQL FILER SOM ER KJØRT FOR ALLE VÆRÅR (1991 - 2020). MÅ FIKSES SLIK AT MAN KAN KJØRE FOR EKSEMPELVIS 5 VÆR ÅR.



# === INITIALIZATIONS ===
START_YEAR = 2005
END_YEAR = 2005

area = None
zone = 'NO1'

# Juster area for å se på sonene, og zone for å se på nodene i sonen
# === COMPUTE TIMERANGE AND PLOT FLOW ===
time_Prod = get_time_steps_for_period(START_YEAR, END_YEAR)
correct_date_start_Prod = DATE_START + pd.Timedelta(hours=time_Prod[0])
if area is not None:
    zones_in_area_prod = getProductionZonesInArea(data, database, area, time_Prod, correct_date_start_Prod, week=True)
if zone is not None:
    nodes_in_zone_prod = getProductionNodesInZone(data, database, zone, time_Prod, correct_date_start_Prod, week=True)



#%% EINAR ###
"""
Production, consumption, and price data for specific nodes within a given time period.

Main Features:
- Handles time using Python's built-in datetime objects.
- Retrieves simulated production, consumption, and price data from a given SQL file for selected nodes within a specified timeframe.
- Organizes data and exports it to an Excel file for further analysis.
"""

# === INITIALIZATIONS ===
START = {"year": 2002, "month": 1, "day": 1, "hour": 0}
END = {"year": 2002, "month": 1, "day": 2, "hour": 0}
Nodes = ['NO4_1', 'NO4_2']
# ======================================================================================================================

start_hour, end_hour = get_hour_range(YEAR_START, YEAR_END, TIMEZONE, START, END)
production_per_node, gen_idx, gen_type = GetProductionAtSpecificNodes(Nodes, data, database, start_hour, end_hour)
consumption_per_node = GetConsumptionAtSpecificNodes(Nodes, data, database, start_hour, end_hour)
nodal_prices_per_node = GetPriceAtSpecificNodes(Nodes, data, database, start_hour, end_hour)
excel_filename = ExportToExcel(Nodes, production_per_node, consumption_per_node, nodal_prices_per_node, START, END, case, version, OUTPUT_PATH)










