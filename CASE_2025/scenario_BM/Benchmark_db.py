from functions.work_functions import *
from functions.global_functions import *  # Functions like 'read_grid_data', 'solve_lp' m.m.
from functions.database_functions import  * # Functions like 'getSystemCostFromDB' m.m.
from zoneinfo import ZoneInfo
from powergama.database import Database  # Import Database-Class specifically
import pandas as pd
import numpy as np


# === General Configurations ===
SIM_YEAR_START = 1991           # Start year for the main simulation  (SQL-file)
SIM_YEAR_END = 2020             # End year for the main simulation  (SQL-file)
CASE_YEAR = 2025
SCENARIO = 'BM'
VERSION = 'v130_sens'
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


# %% Nordic Grid Map

# === INITIALIZATIONS ===
START = {"year": 1991, "month": 1, "day": 1, "hour": 0}
END = {"year": 2020, "month": 12, "day": 31, "hour": 23}
nordic_grid_map_fromDB(data, database, time_range = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END),
                       OUTPUT_PATH = OUTPUT_PATH, version = VERSION, START = START, END = END, exchange_rate_NOK_EUR = 11.38)


# %% === ZONAL PRICE MAP ===
# TODO: legg til mulighet for å ha øre/kwh
zones = ['NO1', 'NO2', 'NO3', 'NO4', 'NO5', 'SE1', 'SE2', 'SE3', 'SE4',
         'DK1', 'DK2', 'FI', 'DE', 'GB', 'NL', 'LT', 'PL', 'EE']
year_range = list(range(SIM_YEAR_START, SIM_YEAR_END + 1))
price_matrix, log = createZonePriceMatrix(data, database, zones, year_range, TIMEZONE, SIM_YEAR_START, SIM_YEAR_END)
# Plot Zonal Price Matrix
plotZonePriceMatrix(price_matrix, save_fig=True, OUTPUT_PATH_PLOTS=OUTPUT_PATH_PLOTS, start=SIM_YEAR_START, end=SIM_YEAR_END, version=VERSION)


# %% Check Total Consumption for a given period.
# Demand Response
# === INITIALIZATIONS ===
START = {"year": 2002, "month": 1, "day": 1, "hour": 0}
END = {"year": 2002, "month": 12, "day": 31, "hour": 23}

time_Demand = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
demandTotal = getDemandPerAreaFromDB(data, database, area='NO', timeMaxMin=time_Demand)
print(sum(demandTotal['sum']))


# %% === Get Production Data ===
# === INITIALIZATIONS ===
START = {"year": 2020, "month": 1, "day": 1, "hour": 0}
END = {"year": 2020, "month": 12, "day": 31, "hour": 23}
area = 'DK'

time_Prod = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
total_Production = getProductionPerAreaFromDB(data, database, time_Prod, area)
print(total_Production)



# %% Collect the system cost and mean area price for the system for a given period

# === INITIALIZATIONS ===
START = {"year": 1991, "month": 1, "day": 1, "hour": 0}
END = {"year": 2020, "month": 12, "day": 31, "hour": 23}

time_SC_MP = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
calcSystemCostAndMeanPriceFromDB(data, database, time_SC_MP, time_SC_MP)


# %% Map prices and branch utilization

# === INITIALIZATIONS ===
START = {"year": 2009, "month": 1, "day": 1, "hour": 0}
END = {"year": 2010, "month": 12, "day": 31, "hour": 0}

time_Map = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
plot_Map(data, database, time_Map, DATE_START, OUTPUT_PATH, version)


# %% GET FLOW ON CHOSEN BRANCHES

# === INITIALIZATIONS ===
START = {"year": 2020, "month": 1, "day": 1, "hour": 0}
END = {"year": 2020, "month": 12, "day": 31, "hour": 23}

# === PLOT CONFIGURATIONS ===

plot_config = {
    "plot_by_year": False,      # Each year in individual plot or all years collected in one plot
    "duration_curve": False,     # True: Plot duration curve, or False: Plot storage filling over time
    "duration_relative": False, # Hours(False) or Percentage(True)
    "save_fig": False,          # True: Save plot as pdf
    "interval": 1,              # Number of months on x-axis. 1 = Step is one month, 12 = Step is 12 months
    "check": False,
    "tex_font": False
}

# === CHOOSE BRANCHES TO CHECK ===
SELECTED_BRANCHES  = [['NO1_5', 'SE3_5'],['NO3_1','SE2_4'], ['NO4_3','SE2_1'], ['NO4_1','SE1_1']] # See branch CSV files for correct connections

# === COMPUTE TIMERANGE AND PLOT FLOW ===
time_Lines = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
plot_Flow_fromDB(data, database, DATE_START, time_Lines, OUTPUT_PATH_PLOTS, plot_config, SELECTED_BRANCHES)

# %% PLOT STORAGE FILLING FOR AREAS

# === INITIALIZATIONS ===
START = {"year": 1991, "month": 1, "day": 1, "hour": 0}
END = {"year": 2020, "month": 12, "day": 31, "hour": 23}

# === PLOT CONFIGURATIONS ===
#  TODO: NÅR SAVE_FIG = TRUE --> DENNE BLIR SVG, IKKE PDF
plot_config = {
    'areas': ['NO'],            # When plotting multiple years in one year, recommend to only use one area
    'relative': True,           # Relative storage filling, True gives percentage
    "plot_by_year": True,       # True: One curve for each year in same plot, or False:all years collected in one plot over the whole simulation period
    "duration_curve": False,    # True: Plot duration curve, or False: Plot storage filling over time
    "save_fig": False,          # True: Save plot as pdf
    "interval": 1,              # Number of months on x-axis. 1 = Step is one month, 12 = Step is 12 months
    "empty_threshold": 1e-6     # If relative (True), then empty threshold is in percentage. If relative (False), then empty threshold is in MWh.
}

# === COMPUTE TIMERANGE AND PLOT FLOW ===
time_SF = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
plot_SF_Areas_FromDB(data, database, time_SF, OUTPUT_PATH_PLOTS, DATE_START, plot_config)

# %% PLOT STORAGE FILLING ZONES
# Todo: Trengs det fortsatt litt jobb med scaleringen av selve plottet, men det er ikke krise enda.
# Todo: Må OGSÅ ha mulighet til å plotte storage filling ned på node nivå.

# === INITIALIZATIONS ===
START = {"year": 1991, "month": 1, "day": 1, "hour": 0}
END = {"year": 2020, "month": 12, "day": 31, "hour": 23}

# === PLOT CONFIGURATIONS ===
plot_config = {
    'zones': ['NO3'],                     # When plotting multiple years in one year, recommend to only use one zone
    'relative': True,                            # Relative storage filling, True gives percentage
    "plot_by_year": 3,                           # (1) Each year in individual plot, (2) Entire Timeline, (3) Each year show over 1 year timeline.
    "duration_curve": False,                     # True: Plot duration curve, or False: Plot storage filling over time
    "save_fig": False,                           # True: Save plot as pdf
    "interval": 1,                                # Number of months on x-axis. 1 = Step is one month, 12 = Step is 12 months
    "empty_threshold": 1e-6                         # If relative (True), then empty threshold is in percentage. If relative (False), then empty threshold is in MWh.
}

# === COMPUTE TIMERANGE AND PLOT FLOW ===
time_SF = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
plot_SF_Zones_FromDB(data, database, time_SF, OUTPUT_PATH_PLOTS, DATE_START, plot_config)


# %% Plot nodal prices Norway in a zone

#Todo: Denne fungerer ikke:"plot_by_year": True,  # Each year in individual plot or all years collected in one plot

# === INITIALIZATIONS ===
START = {"year": 1992, "month": 1, "day": 1, "hour": 0}
END = {"year": 1993, "month": 1, "day": 1, "hour": 0}

# === PLOT CONFIGURATIONS ===
plot_config = {
    'zone': 'NO2',                          # When plotting multiple years in one year, recommend to only use one zone
    'plot_all_nodes': True,                 # (True) Plot all nodes in a zone or (False) avg of all nodes
    "plot_by_year": False,                  # (True) Each year in individual plot or all years collected in one plot
    "duration_curve": False,                # (True) Plot duration curve, or (False) Plot storage filling over time PRICE OVER TIME?
    "save_fig": False,                      # True: Save plot as pdf
    "interval": 1,                          # Number of months on x-axis. 1 = Step is one month, 12 = Step is 12 months
    "tex_font": False

}


# === COMPUTE TIMERANGE AND PLOT PRICE ===
time_NP = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
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


# === COMPUTE TIMERANGE AND PLOT PRICE ===
time_ZP = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
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
time_HRI = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
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
time_PLP = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
calcPlot_PLP_FromDB(data, database, time_PLP, OUTPUT_PATH_PLOTS, DATE_START, plot_config)



# %% Load, generation by type in AREA

# === INITIALIZATIONS ===
START = {"year": 1992, "month": 1, "day": 1, "hour": 0}
END = {"year": 1993, "month": 6, "day": 1, "hour": 0}


# === PLOT CONFIGURATIONS ===
plot_config = {
    'area': 'PL',
    'title': 'Production, Consumption and Price in NO',
    "fig_size": (10, 6),
    "plot_full_timeline": True,
    "duration_curve": False,
    "box_in_frame": False,
    "save_fig": False,                      # True: Save plot as pdf
    "interval": 1                           # Number of months on x-axis. 1 = Step is one month, 12 = Step is 12 months
}


# === COMPUTE TIMERANGE AND PLOT FLOW ===
time_LGT = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
df_gen_re, df_prices, tot_prod = calcPlot_LG_FromDB(data, database, time_LGT, OUTPUT_PATH_PLOTS, DATE_START, plot_config)
print(f"Total production in {plot_config['area']}: {tot_prod:.2f} MWh")


# %% Production data for area or zone

""" Initialize data for Production"""
# La area eller zone være None om de ikke skal brukes.

# LAGT PÅ IS

# === INITIALIZATIONS ===
START = {"year": 1992, "month": 1, "day": 1, "hour": 0}
END = {"year": 1993, "month": 1, "day": 1, "hour": 0}
area = None
zone = 'NO1'

# Juster area for å se på sonene, og zone for å se på nodene i sonen
# === COMPUTE TIMERANGE AND PLOT FLOW ===
time_Prod = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
correct_date_start_Prod = DATE_START + pd.Timedelta(hours=time_Prod[0])
if area is not None:
    zones_in_area_prod = getProductionZonesInArea(data, database, area, time_Prod, correct_date_start_Prod, week=True)
if zone is not None:
    nodes_in_zone_prod = getProductionNodesInZone(data, database, zone, time_Prod, correct_date_start_Prod, week=True)


# === EINAR ===
# %% National-level electricity production and consumption
"""
Retrieves and aggregates electricity production and demand data at the national level.

Overview:
- Extracts simulated weekly production and consumption data for a specified country and time period.
- Organizes the output into two levels of temporal resolution:
    - A weekly-level DataFrame indexed by datetime.
    - An annual-level DataFrame obtained by summing values across calendar years.
"""

# === INITIALIZATIONS ===
START = {"year": 1991, "month": 1, "day": 1, "hour": 0}
END = {"year": 1991, "month": 12, "day": 31, "hour": 23}

df_gen_dem, df_prices, total = get_production_by_type_FromDB(data, database, "NO", get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END), "1991-01-01")
df_gen_dem.index = pd.to_datetime(df_gen_dem.index)
df_gen_dem['Year'] = df_gen_dem.index.year
df_gen_yearly = df_gen_dem.groupby('Year').sum()


# %% TETS - NY APPROACH BASERT PÅ TIDSSTEG OG IKKE DATO! National-level electricity production and consumption
"""
Retrieves and aggregates electricity production and demand data at the national level.

Based on idealyears over the 30-year climate periode so that each year has same number of hours so it can be comparable to each other.


Overview:

"""

# === INITIALIZATIONS ===
country = "FI"  # Country code

n_ideal_years = 30
n_timesteps = int(8766.4 * n_ideal_years) # Ved full 30-års simuleringsperiode


df_gen, df_prices, total_production, df_gen_per_year = get_production_by_type_ideal_timestep(
    data=data,
    db=database,
    area_OP=country,
    n_timesteps=n_timesteps
)



# %% Node-level production by type
"""
Retrieval and aggregation of electricity production by technology type at the node level 
for a specified time period.

Overview:
- Calculates the total electricity production (in MWh) by each production type for selected nodes.
- Accepts a list of nodes and a time interval as input.
- Queries simulation results from a structured SQL database.
- Returns a structured DataFrame where rows represent nodes and columns represent production types.

"""

# === INITIALIZATIONS ===
START = {"year": 1991, "month": 1, "day": 1, "hour": 0}
END = {"year": 1991, "month": 12, "day": 31, "hour": 23}
nodes = ["DK1_2", "DK1_3", "DK2_2"]

time_range = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
dispatch_df = get_total_production_by_type_per_node(data, database, nodes, time_range)

# %% Capture price

#
# # === INITIALIZATIONS ===
# START = {"year": 1998, "month": 1, "day": 1, "hour": 0}
# END = {"year": 1998, "month": 12, "day": 31, "hour": 23}
# nodes = ["DK1_2", "DK1_3", "DK2_2"]
#
# start_hour, end_hour = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
# production_per_node, gen_idx, gen_type = GetProductionAtSpecificNodes(nodes, data, database, start_hour, end_hour)
# nodal_prices_per_node = GetPriceAtSpecificNodes(nodes, data, database, start_hour, end_hour)
#
# capture_prices_df, capture_rates_df = CalculateCapturePrice(production_per_node, nodal_prices_per_node)

# %% Capture price

# === INITIALIZATIONS ===
START = {"year": 1991}
END = {"year": 2020}
SELECTED_NODES = ["DK1_2", "DK1_3", "DK2_2"]
# SELECTED_NODES = "ALL"
nodes = data.node["id"].dropna().unique().tolist() if SELECTED_NODES == "ALL" else SELECTED_NODES
capture_prices_over_years, value_factors_over_years = CalculateCapturePriceAndValueFactorOverYears(START, END, nodes, data=data, database=database, timezone=TIMEZONE)


# %% Sensitivities Nuclear production

# === INITIALIZATIONS ===
START = {"year": 1991, "month": 1, "day": 1, "hour": 0}
END = {"year": 2020, "month": 12, "day": 31, "hour": 23}

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


#%% Excel ###
"""
Production, consumption, and price data for specific nodes within a given time period.

Main Features:
- Handles time using Python's built-in datetime objects.
- Retrieves simulated production, consumption, and price data from a given SQL file for selected nodes within a specified timeframe.
- Organizes data and exports it to an Excel file for further analysis.
"""

# === INITIALIZATIONS ===
START = {"year": 2020, "month": 1, "day": 1, "hour": 0}
END = {"year": 2020, "month": 12, "day": 31, "hour": 23}
Nodes = ["DK1_2", "DK1_3", "DK2_2"]

SELECTED_BRANCHES  = None
# ======================================================================================================================

start_hour, end_hour = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
production_per_node, gen_idx, gen_type = GetProductionAtSpecificNodes(Nodes, data, database, start_hour, end_hour)
consumption_per_node = GetConsumptionAtSpecificNodes(Nodes, data, database, start_hour, end_hour)
nodal_prices_per_node = GetPriceAtSpecificNodes(Nodes, data, database, start_hour, end_hour)
reservoir_filling_per_node, storage_cap = GetReservoirFillingAtSpecificNodes(Nodes, data, database, start_hour, end_hour)
flow_data = getFlowDataOnBranches(data, database, [start_hour, end_hour], SELECTED_BRANCHES)
excel_filename = ExportToExcel(Nodes, production_per_node, consumption_per_node, nodal_prices_per_node, reservoir_filling_per_node, storage_cap, flow_data, START, END, SCENARIO, VERSION, OUTPUT_PATH)



# %% === Write Flow Data to Excel ===

# === INITIALIZATIONS ===
START = {"year": 1993, "month": 1, "day": 1, "hour": 0}
END = {"year": 1994, "month": 1, "day": 1, "hour": 0}

# === CHOOSE BRANCHES TO CHECK ===
SELECTED_BRANCHES  = [['NO3_1','SE2_4'],['NO3_1','NO4_3'], ['NO3_4','NO5_1'], ['NO3_5','NO1_1']] # See branch CSV files for correct connections

time_Flow = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
flow_data = getFlowDataOnBranches(database, time_Flow, GRID_DATA_PATH, SELECTED_BRANCHES)
flow_path = writeFlowToExcel(flow_data, START, END, TIMEZONE, OUTPUT_PATH, SCENARIO, VERSION)


# %% === Get production by type aggregate by zone ===
'''
Aggregates electricity production by zone and production type over a specified time period. 
Returns two DataFrames: (1) detailed production per type, and (2) production merged into broader categories, in TWh.

Input:
- SELECTED_NODES = List of node IDs (e.g., ["NO1_1", "NO1_2", "NO1_3"]) 
  or the string "ALL" to include all nodes in the system.
'''
# === INITIALIZATIONS ===
START = {"year": 2020, "month": 1, "day": 1, "hour": 0}
END = {"year": 2020, "month": 12, "day": 31, "hour": 23}
SELECTED_NODES = "ALL"
# SELECTED_NODES = ["SE4_1", "SE4_2", "SE3_1", "SE2_1", "SE2_2"]
# ======================================================================================================================
zone_summed_df, zone_summed_merged_df = get_zone_production_summary(SELECTED_NODES, START, END, TIMEZONE, SIM_YEAR_START, SIM_YEAR_END, data, database)


# %% === For validation of simulated production per. type in each zone against historical data. ===

# Klargjør simuleringsresultat for validation mot eSett Open Data
zone_summed_merged_df["Other"] = 0
zone_summed_merged_df = zone_summed_merged_df[[col for col in zone_summed_merged_df.columns if col != "Other"] + ["Other"]]
zones_to_remove = ["GB", "DE", "NL", "EE", "LT", "PL"]
zone_summed_merged_df = zone_summed_merged_df.drop(index=zones_to_remove, errors="ignore")


# Data from eSett Open Data (2024) and energinet manually structured. (OBS: Other for DK is not correct)
data = {
    "Production total": [
        "23 504 976,88", "10 415 509,92", "77 559 009,40", "21 655 091,37", "55 614 115,20",
        "23 531 833,16", "23 831 200,10", "32 486 028,55", "24 340 088,61", "52 568 108,96",
        "75 764 368,13", "9 266 869,76"
    ],
    "Hydro": [
        "0,00", "0,00", "14 163 404,94", "20 373 792,38", "50 772 720,95",
        "17 614 838,05", "18 925 248,00", "32 295 294,09", "16 742 020,14", "34 653 435,57",
        "11 558 514,61", "1 435 190,32"
    ],
    "Nuclear": [
        "0,00", "0,00", "31 089 513,50", "0,00", "0,00",
        "0,00", "0,00", "0,00", "0,00", "0,00",
        "48 739 158,76", "0,00"
    ],
    "Solar": [
        "2621978,62", "1077359,66", "372 558,49", "112 435,12", "96 500,88",
        "13 520,87", "712,29", "16 400,42", "14 033,40", "88 709,92",
        "1 539 778,89", "820 842,69"
    ],
    "Thermal": [
        "5896092,701", "4440883,971", "10 097 294,82", "210 522,57", "185 222,31",
        "164 244,14", "1 540 310,13", "174 115,91", "135 590,53", "785 127,05",
        "3 363 138,03", "1 115 343,87"
    ],
    "Wind Onshore": [
        "9165290,836", "6129726,786", "20 054 297,31", "957 151,56", "4 557 816,79",
        "5 738 724,92", "3 363 439,44", "2,25", "7 447 713,88", "17 029 707,54",
        "10 471 969,52", "5 873 242,13"
    ],
    "Wind Offshore": [
        "6129726,786", "3605418,939", "0,00", "0,00", "0,00",
        "0,00", "0,00", "0,00", "0,00", "0,00",
        "0,00", "0,00"
    ],
    "Other": [
        "0", "0", "1 779 636,23", "1 189,74", "1 854,27",
        "505,19", "1 490,23", "215,88", "730,66", "11 128,88",
        "89 080,12", "22 250,75"
    ]
}

# Define index (zone names)
index = [
    "DK1", "DK2", "FI", "NO1", "NO2", "NO3", "NO4", "NO5",
    "SE1", "SE2", "SE3", "SE4"
]

df_eSett = pd.DataFrame(data, index=index)
df_eSett = df_eSett.apply(lambda col: col.map(lambda x: float(str(x).replace(" ", "").replace(",", "."))))
df_eSett = df_eSett / 1e6
# comparison_df = zone_summed_merged_df - df_eSett
# percent_error_df = 100 * (zone_summed_merged_df - df_eSett) / df_eSett

# Ensure both DataFrames have the same row and column order
zone_summed_merged_df = zone_summed_merged_df.reindex(index=df_eSett.index, columns=df_eSett.columns)

# 1. Calculate absolute difference
diff_df = zone_summed_merged_df - df_eSett

# 2. Calculate percent error (relative difference)
percent_error_df = 100 * (zone_summed_merged_df - df_eSett) / df_eSett

# Optional: round to 2 decimal places for neat display
diff_df = diff_df.round(2)
percent_error_df = percent_error_df.round(2)

# Print or display
print("Absolute difference (TWh):")
print(diff_df)

print("\nPercent error (%):")
print(percent_error_df)

# %% FRA baseline

START = {"year": 2020, "month": 1, "day": 1, "hour": 0}
END = {"year": 2020, "month": 12, "day": 31, "hour": 23}
time_EB = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
all_nodes = data.node.id

flow_data = getFlowDataOnALLBranches(data, database, time_EB)       # TAR LANG TID
totalDemand = getDemandForAllNodesFromDB(data, database, time_EB)
totalProduction = getProductionForAllNodesFromDB(data, database, time_EB)   # TAR LANG TID
totalLoadShedding = database.getResultLoadheddingSum(timeMaxMin=time_EB)
node_energyBalance = getEnergyBalanceNodeLevel(all_nodes, totalDemand, totalProduction, totalLoadShedding, flow_data, OUTPUT_PATH, VERSION, START)
zone_energyBalance = getEnergyBalanceZoneLevel(all_nodes, totalDemand, totalProduction, totalLoadShedding, flow_data, OUTPUT_PATH, VERSION, START)


# %% FRA baseline
START = {"year": 2020, "month": 1, "day": 1, "hour": 0}
END = {"year": 2020, "month": 12, "day": 31, "hour": 23}
time_period = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)

save_production_to_excel(data, database, time_period, START, END, TIMEZONE, OUTPUT_PATH / 'data_files', VERSION)