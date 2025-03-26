from functions.work_functions import *
from functions.global_functions import *  # Functions like 'read_grid_data', 'solve_lp' m.m.
from functions.database_functions import  * # Functions like 'getSystemCostFromDB' m.m.
from zoneinfo import ZoneInfo
from powergama.database import Database  # Import Database-Class specifically
import pandas as pd

#midlertidig:

from datetime import datetime, timedelta
from powergama.GridData import GridData
from openpyxl import Workbook, load_workbook
from openpyxl.chart import LineChart, Reference
import os
from openpyxl import Workbook
from openpyxl.chart import BarChart, LineChart, Reference


# === General Configurations ===
YEAR_SCENARIO = 2025
SIM_YEAR_START = 1991           # Start year for the main simulation  (SQL-file)
SIM_YEAR_END = 2020             # End year for the main simulation  (SQL-file)
case = 'BM'
version = 'v83'
TIMEZONE = ZoneInfo("UTC")  # Definerer UTC tidssone


DATE_START = pd.Timestamp(f'{SIM_YEAR_START}-01-01 00:00:00', tz='UTC')
DATE_END = pd.Timestamp(f'{SIM_YEAR_END}-12-31 23:00:00', tz='UTC')

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
SQL_FILE = BASE_DIR / f"powergama_{case}_{version}_{SIM_YEAR_START}_{SIM_YEAR_END}.sqlite"
DATA_PATH = BASE_DIR / 'data'
GRID_DATA_PATH = DATA_PATH / 'system'
OUTPUT_PATH = BASE_DIR / 'results'
OUTPUT_PATH_PLOTS = BASE_DIR / 'results' / 'plots'

# === Initialize Database and Grid Data ===
data, time_max_min = setup_grid(YEAR_SCENARIO, version, DATE_START, DATE_END, DATA_PATH, new_scenario, save_scenario, case)
database = Database(SQL_FILE)


# %% Nordic Grid Map

# === INITIALIZATIONS ===
START = {"year": 1991, "month": 1, "day": 1, "hour": 0}
END = {"year": 2020, "month": 12, "day": 31, "hour": 23}
nordic_grid_map_fromDB(data, database, time_range = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END),
                       OUTPUT_PATH = OUTPUT_PATH, version = version, START = START, END = END, exchange_rate_NOK_EUR = 11.38)


# %% === ZONAL PRICE MAP ===

zones = ['NO1', 'NO2', 'NO3', 'NO4', 'NO5', 'SE1', 'SE2', 'SE3', 'SE4',
         'DK1', 'DK2', 'FI', 'DE', 'GB', 'NL', 'LT', 'PL', 'EE']
year_range = list(range(SIM_YEAR_START, SIM_YEAR_END + 1))
price_matrix = createZonePriceMatrix(data, database, zones, year_range, TIMEZONE, SIM_YEAR_START, SIM_YEAR_END)

# %% Plot Zonal Price Matrix
plotZonePriceMatrix(price_matrix, save_fig=True, OUTPUT_PATH_PLOTS=OUTPUT_PATH_PLOTS)


# %% Collect the system cost and mean area price for the system for a given period
#TODO: TypeError: calcSystemCostAndMeanPriceFromDB() takes 4 positional arguments but 5 were given

# === INITIALIZATIONS ===
START = {"year": 1992, "month": 1, "day": 1, "hour": 0}
END = {"year": 1993, "month": 1, "day": 1, "hour": 0}

time_SC = get_hour_range(YEAR_START, YEAR_END, TIMEZONE, START, END)
time_MP = get_hour_range(YEAR_START, YEAR_END, TIMEZONE, START, END)
calcSystemCostAndMeanPriceFromDB(data, database, time_SC, time_MP)


# %% Map prices and branch utilization

# === INITIALIZATIONS ===
START = {"year": 2009, "month": 1, "day": 1, "hour": 0}
END = {"year": 2010, "month": 12, "day": 31, "hour": 0}

time_Map = get_hour_range(YEAR_START, YEAR_END, TIMEZONE, START, END)
plot_Map(data, database, time_Map, DATE_START, OUTPUT_PATH, version)


# %% GET FLOW ON CHOSEN BRANCHES

# === INITIALIZATIONS ===
START = {"year": 1992, "month": 1, "day": 1, "hour": 0}
END = {"year": 1993, "month": 1, "day": 1, "hour": 0}

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
SELECTED_BRANCHES  = [['NO3_1','SE2_4'],['NO3_1','NO4_3'], ['NO3_4','NO5_1'], ['NO3_5','NO1_1']] # See branch CSV files for correct connections

# === COMPUTE TIMERANGE AND PLOT FLOW ===
time_Lines = get_hour_range(YEAR_START, YEAR_END, TIMEZONE, START, END)
plot_Flow_fromDB(database, DATE_START, time_Lines, GRID_DATA_PATH, OUTPUT_PATH_PLOTS, plot_config, SELECTED_BRANCHES)



# %% PLOT STORAGE FILLING FOR AREAS

# === INITIALIZATIONS ===
START = {"year": 1991, "month": 1, "day": 1, "hour": 0}
END = {"year": 2020, "month": 12, "day": 31, "hour": 23}

# === PLOT CONFIGURATIONS ===
plot_config = {
    'areas': ['NO'],            # When plotting multiple years in one year, recommend to only use one area
    'relative': True,           # Relative storage filling, True gives percentage
    "plot_by_year": True,       # True: One curve for each year in same plot, or False:all years collected in one plot over the whole simulation period
    "duration_curve": False,    # True: Plot duration curve, or False: Plot storage filling over time
    "save_fig": False,          # True: Save plot as pdf
    "interval": 1               # Number of months on x-axis. 1 = Step is one month, 12 = Step is 12 months
}

# === COMPUTE TIMERANGE AND PLOT FLOW ===
time_SF = get_hour_range(YEAR_START, YEAR_END, TIMEZONE, START, END)
plot_SF_Areas_FromDB(data, database, time_max_min, OUTPUT_PATH_PLOTS, DATE_START, plot_config)

# %% PLOT STORAGE FILLING ZONES
# Todo: Trengs det fortsatt litt jobb med scaleringen av selve plottet, men det er ikke krise enda.
# Todo: Må OGSÅ ha mulighet til å plotte storage filling ned på node nivå.

# === INITIALIZATIONS ===
START = {"year": 1992, "month": 1, "day": 1, "hour": 0}
END = {"year": 1993, "month": 1, "day": 1, "hour": 0}

# === PLOT CONFIGURATIONS ===
plot_config = {
    'zones': ['NO2', 'NO3'],                     # When plotting multiple years in one year, recommend to only use one zone
    'relative': True,                            # Relative storage filling, True gives percentage
    "plot_by_year": 1,                           # (1) Each year in individual plot, (2) Entire Timeline, (3) Each year show over 1 year timeline.
    "duration_curve": False,                     # True: Plot duration curve, or False: Plot storage filling over time
    "save_fig": False,                           # True: Save plot as pdf
    "interval": 1                                # Number of months on x-axis. 1 = Step is one month, 12 = Step is 12 months
}

# If you want to go in and change title, follow the function from here to its source location and change it there.
# Remember that you then have to reset the console run
# === COMPUTE TIMERANGE AND PLOT FLOW ===
time_SF = get_hour_range(YEAR_START, YEAR_END, TIMEZONE, START, END)
plot_SF_Zones_FromDB(data, database, time_SF, OUTPUT_PATH_PLOTS, DATE_START, plot_config)


# %% Plot nodal prices Norway in a zone

#Todo: Denne fungerer ikke:"plot_by_year": True,  # Each year in individual plot or all years collected in one plot

# === INITIALIZATIONS ===
START = {"year": 1992, "month": 1, "day": 1, "hour": 0}
END = {"year": 1993, "month": 1, "day": 1, "hour": 0}

# === PLOT CONFIGURATIONS ===
plot_config = {
    'zone': 'NO3',                          # When plotting multiple years in one year, recommend to only use one zone
    'plot_all_nodes': True,                 # (True) Plot all nodes in a zone or (False) avg of all nodes
    "plot_by_year": False,                  # (True) Each year in individual plot or all years collected in one plot
    "duration_curve": False,                # (True) Plot duration curve, or (False) Plot storage filling over time PRICE OVER TIME?
    "save_fig": False,                      # True: Save plot as pdf
    "interval": 1,                          # Number of months on x-axis. 1 = Step is one month, 12 = Step is 12 months
    "tex_font": False

}

# === COMPUTE TIMERANGE AND PLOT FLOW ===
time_NP = get_hour_range(YEAR_START, YEAR_END, TIMEZONE, START, END)
calcPlot_NP_FromDB(data, database, time_NP, OUTPUT_PATH_PLOTS, DATE_START, plot_config)


# %% PLOT ZONAL PRICES

# === INITIALIZATIONS ===
START = {"year": 1992, "month": 1, "day": 1, "hour": 0}
END = {"year": 1993, "month": 1, "day": 1, "hour": 0}

# === PLOT CONFIGURATIONS ===
plot_config = {
    'zones': ['NO5'],                       # Zones for plotting
    "plot_by_year": True,                   # (True)Each year in individual plot or (False) all years collected in one plot
    "duration_curve": False,                # True: Plot duration curve, or False: Plot storage filling over time
    "save_fig": False,                      # True: Save plot as pdf
    "interval": 1,                          # Number of months on x-axis. 1 = Step is one month, 12 = Step is 12 months
    "tex_font": False                       # Keep false
}

# === COMPUTE TIMERANGE AND PLOT FLOW ===
time_ZP = get_hour_range(YEAR_START, YEAR_END, TIMEZONE, START, END)
calcPlot_ZonalPrices_FromDB(data, database, time_ZP, OUTPUT_PATH_PLOTS, DATE_START, plot_config)



# %% Hydro production, reservoir filling, inflow

# === INITIALIZATIONS ===
START = {"year": 1992, "month": 1, "day": 1, "hour": 0}
END = {"year": 1993, "month": 1, "day": 1, "hour": 0}

# === PLOT CONFIGURATIONS ===
plot_config = {
    'area': 'NO',
    'genType': 'hydro',
    'relative_storage': True,               # (True) percentage, or (False) real values
    "plot_full_timeline": False,            # (True) plot the full timeline or (False) plot by year.
    "box_in_frame": False,                  # (True) legend box in frame or (False) outside frame
    "save_fig": False,                      # True: Save plot as pdf
    "interval": 1                           # Number of months on x-axis. 1 = Step is one month, 12 = Step is 12 months
}

# === COMPUTE TIMERANGE AND PLOT FLOW ===
time_HRI = get_hour_range(YEAR_START, YEAR_END, TIMEZONE, START, END)
calcPlot_HRI_FromDB(data, database, time_HRI, OUTPUT_PATH_PLOTS, DATE_START, plot_config)


# %% Plot nodal prices, demand and hydro production

# === INITIALIZATIONS ===
START = {"year": 1992, "month": 1, "day": 1, "hour": 0}
END = {"year": 1993, "month": 1, "day": 1, "hour": 0}

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
time_PLP = get_hour_range(YEAR_START, YEAR_END, TIMEZONE, START, END)
calcPlot_PLP_FromDB(data, database, time_PLP, OUTPUT_PATH_PLOTS, DATE_START, plot_config)

# %% Check Total Consumption for a given period.
# Demand Response
# === INITIALIZATIONS ===
START = {"year": 2010, "month": 1, "day": 1, "hour": 0}
END = {"year": 2011, "month": 1, "day": 1, "hour": 0}

time_Demand = get_hour_range(YEAR_START, YEAR_END, TIMEZONE, START, END)
demandTotal = getDemandPerAreaFromDB(data, database, area='NO', timeMaxMin=time_Demand)
print(sum(demandTotal['sum']))


# %% === Get Production Data ===
# === INITIALIZATIONS ===
START = {"year": 1999, "month": 1, "day": 1, "hour": 0}
END = {"year": 1999, "month": 1, "day": 2, "hour": 0}
area = 'NO'

time_Prod = get_hour_range(YEAR_START, YEAR_END, TIMEZONE, START, END)
total_Production = getProductionPerAreaFromDB(data, database, time_Prod, area)
print(total_Production)

# %% Load, generation by type in AREA

# === INITIALIZATIONS ===
START = {"year": 1992, "month": 1, "day": 1, "hour": 0}
END = {"year": 1993, "month": 1, "day": 1, "hour": 0}



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
time_LGT = get_hour_range(YEAR_START, YEAR_END, TIMEZONE, START, END)
df_gen_re, df_prices, tot_prod = calcPlot_LG_FromDB(data, database, time_LGT, OUTPUT_PATH_PLOTS, DATE_START, plot_config)
print(f"Total production in {plot_config['area']}: {tot_prod:.2f} MWh")


# %% Production data for area or zone

""" Initialize data for Production"""
# La area eller zone være None om de ikke skal brukes.

# TODO: DENNE FUNGERER KUN FOR SQL FILER SOM ER KJØRT FOR ALLE VÆRÅR (1991 - 2020). MÅ FIKSES SLIK AT MAN KAN KJØRE FOR EKSEMPELVIS 5 VÆR ÅR.



# === INITIALIZATIONS ===
START = {"year": 1992, "month": 1, "day": 1, "hour": 0}
END = {"year": 1993, "month": 1, "day": 1, "hour": 0}
area = None
zone = 'NO1'

# Juster area for å se på sonene, og zone for å se på nodene i sonen
# === COMPUTE TIMERANGE AND PLOT FLOW ===
time_Prod = get_hour_range(YEAR_START, YEAR_END, TIMEZONE, START, END)
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
START = {"year": 1992, "month": 1, "day": 1, "hour": 0}
END = {"year": 1993, "month": 1, "day": 1, "hour": 0}
Nodes = ['NO5_1']
SELECTED_BRANCHES  = [['NO3_1','SE2_4'],['NO3_1','NO4_3']] # See branch CSV files for correct connections
# ======================================================================================================================

start_hour, end_hour = get_hour_range(YEAR_START, YEAR_END, TIMEZONE, START, END)
production_per_node, gen_idx, gen_type = GetProductionAtSpecificNodes(Nodes, data, database, start_hour, end_hour)
consumption_per_node = GetConsumptionAtSpecificNodes(Nodes, data, database, start_hour, end_hour)
nodal_prices_per_node = GetPriceAtSpecificNodes(Nodes, data, database, start_hour, end_hour)
reservoir_filling_per_node, storage_cap = GetReservoirFillingAtSpecificNodes(Nodes, data, database, start_hour, end_hour)
flow_data = getFlowDataOnBranches(database, [start_hour, end_hour], GRID_DATA_PATH, SELECTED_BRANCHES)
excel_filename = ExportToExcel(Nodes, production_per_node, consumption_per_node, nodal_prices_per_node, reservoir_filling_per_node, storage_cap, flow_data, START, END, case, version, OUTPUT_PATH)


# %% === Write Flow Data to Excel ===

# === INITIALIZATIONS ===
START = {"year": 1993, "month": 1, "day": 1, "hour": 0}
END = {"year": 1994, "month": 1, "day": 1, "hour": 0}

# === CHOOSE BRANCHES TO CHECK ===
SELECTED_BRANCHES  = [['NO3_1','SE2_4'],['NO3_1','NO4_3'], ['NO3_4','NO5_1'], ['NO3_5','NO1_1']] # See branch CSV files for correct connections

time_Flow = get_hour_range(YEAR_START, YEAR_END, TIMEZONE, START, END)
flow_data = getFlowDataOnBranches(database, time_Flow, GRID_DATA_PATH, SELECTED_BRANCHES)
flow_path = writeFlowToExcel(flow_data, START, END, TIMEZONE, OUTPUT_PATH, case, version)









