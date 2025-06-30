from functions.work_functions import *
from functions.global_functions import *  # Functions like 'read_grid_data', 'solve_lp' m.m.
from functions.database_functions import  * # Functions like 'getSystemCostFromDB' m.m.
from zoneinfo import ZoneInfo
from powergama.database import Database  # Import Database-Class specifically
import pandas as pd
from datetime import datetime
import numpy as np


# === General Configurations ===
SIM_YEAR_START = 1991           # Start year for the main simulation  (SQL-file)
SIM_YEAR_END = 2020            # End year for the main simulation  (SQL-file)
CASE_YEAR = 2035
SCENARIO = 'FDT'
VERSION = 'v1'
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
END = {"year": 2020, "month": 12, "day": 31, "hour": 23}
nordic_grid_map_fromDB(data, database, time_range = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END),
                       OUTPUT_PATH = OUTPUT_PATH, version = VERSION, START = START, END = END, exchange_rate_NOK_EUR = 11.38)


# %% === ZONAL PRICE MAP ===
zones = ['NO1', 'NO2', 'NO3', 'NO4', 'NO5', 'SE1', 'SE2', 'SE3', 'SE4',
         'DK1', 'DK2', 'FI', 'DE', 'GB', 'NL', 'LT', 'PL', 'EE']
year_range = list(range(SIM_YEAR_START, SIM_YEAR_END + 1))
price_matrix, log = createZonePriceMatrix(data, database, zones, year_range, TIMEZONE, SIM_YEAR_START, SIM_YEAR_END)
# Plot Zonal Price Matrix
"""
Colormap options:
- 'YlOrRd': Yellow to Red
- 'Blues': Blue shades
- 'Greens': Green shades
- 'Purples': Purple shades
- 'Oranges': Orange shades
- 'Greys': Grey shades
- 'viridis': Viridis colormap
- 'plasma': Plasma colormap
- 'cividis': Cividis colormap
- 'magma': Magma colormap
- 'copper': Copper colormap
- 'coolwarm': Coolwarm colormap
- 'RdBu': Red to Blue colormap
- 'Spectral': Spectral colormap
- 'twilight': Twilight colormap
- 'twilight_shifted': Twilight shifted colormap
- 'cubehelix': Cubehelix colormap
- 'terrain': Terrain colormap
- 'ocean': Ocean colormap
"""
colormap = "plasma"

plotZonePriceMatrix(price_matrix, save_fig=True, OUTPUT_PATH_PLOTS=OUTPUT_PATH_PLOTS, start=SIM_YEAR_START, end=SIM_YEAR_END, version=VERSION, colormap=colormap)


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
plot_SF_Areas_FromDB(data, database, time_SF, OUTPUT_PATH_PLOTS, DATE_START, plot_config, START, END)

# %% STORAGE - PLOT STORAGE FILLING ZONES
# Todo: Trengs det fortsatt litt jobb med scaleringen av selve plottet, men det er ikke krise enda.
# Todo: Må OGSÅ ha mulighet til å plotte storage filling ned på node nivå.

# === INITIALIZATIONS ===
START = {"year": 1991, "month": 1, "day": 1, "hour": 0}
END = {"year": 1993, "month": 12, "day": 31, "hour": 23}

# === PLOT CONFIGURATIONS ===
plot_config = {
    'zones': ['SE3'],               # When plotting multiple years in one year, recommend to only use one zone
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
plot_SF_Zones_FromDB(data, database, time_SF, OUTPUT_PATH_PLOTS, DATE_START, plot_config, START, END)


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


# %% GET FLOW ON CHOSEN BRANCHES

# === INITIALIZATIONS ===
START = {"year": 1991, "month": 1, "day": 1, "hour": 0}
END = {"year": 2020, "month": 12, "day": 31, "hour": 23}

# === PLOT CONFIGURATIONS ===

plot_config = {
    "plot_by_year": False,      # Each year in individual plot or all years collected in one plot
    "duration_curve": True,     # True: Plot duration curve, or False: Plot storage filling over time
    "duration_relative": True, # Hours(False) or Percentage(True)
    "save_fig": True,           # True: Save plot as pdf
    "interval": 1,              # Number of months on x-axis. 1 = Step is one month, 12 = Step is 12 months
    "check": False,
    "tex_font": False
}

# === CHOOSE BRANCHES TO CHECK ===
SELECTED_BRANCHES  = [['NL','NO2_4'],['NO2_4','DE'], ['NO2_1','GB'], ['DK1_1','NO2_5']] # See branch CSV files for correct connections

# === COMPUTE TIMERANGE AND PLOT FLOW ===
time_Lines = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
plot_Flow_fromDB(data, database, DATE_START, time_Lines, OUTPUT_PATH_PLOTS, plot_config, SELECTED_BRANCHES)


# %%# === ENERGY BALANCE ===

energyBalance = {}

for i in range(0, 30):
    year = 1991 + i
    print(year)

    START = {"year": year, "month": 1, "day": 1, "hour": 0}
    END = {"year": year, "month": 12, "day": 31, "hour": 23}
    time_EB = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
    all_nodes = data.node.id

    totalProduction = collectProductionForAllNodesFromDB(data, database, time_EB)
    flow_data = collectFlowDataOnALLBranches(data, database, time_EB)


    #flow_data = getFlowDataOnALLBranches(data, database, time_EB)       # TAR LANG TID
    totalDemand = collectDemandForAllNodesFromDB(data, database, time_EB)
    #totalProduction = getProductionForAllNodesFromDB(data, database, time_EB)   # TAR LANG TID
    totalLoadShedding = database.getResultLoadheddingSum(timeMaxMin=time_EB)

    # Calculate energy balance at node and zone levels
    node_energyBalance = getEnergyBalanceNodeLevel(all_nodes, totalDemand, totalProduction, totalLoadShedding, flow_data, OUTPUT_PATH / 'data_files/energy_balance', VERSION, START)
    zone_energyBalance = getEnergyBalanceZoneLevel(all_nodes, totalDemand, totalProduction, totalLoadShedding, flow_data, OUTPUT_PATH / 'data_files/energy_balance', VERSION, START)
    # Store energy balance results in the dictionary
    energyBalance[year] = {
        "node_level": node_energyBalance,
        "zone_level": zone_energyBalance
    }

# df_importexport = getImportExportFromDB(data, database, timeMaxMin=time_EB) # Import/Export data for all AREAS
# %%
# get average balance in each country
# Initialize dictionary to store yearly sums for each country and metric
sumBalance = {
    'NO': {'Production': [], 'Demand': [], 'Balance_mls': [], 'Balance_uls': []},
    'SE': {'Production': [], 'Demand': [], 'Balance_mls': [], 'Balance_uls': []},
    'FI': {'Production': [], 'Demand': [], 'Balance_mls': [], 'Balance_uls': []},
    'DK': {'Production': [], 'Demand': [], 'Balance_mls': [], 'Balance_uls': []},
    'GB': {'Production': [], 'Demand': [], 'Balance_mls': [], 'Balance_uls': []},
    'DE': {'Production': [], 'Demand': [], 'Balance_mls': [], 'Balance_uls': []},
    'NL': {'Production': [], 'Demand': [], 'Balance_mls': [], 'Balance_uls': []},
    'LT': {'Production': [], 'Demand': [], 'Balance_mls': [], 'Balance_uls': []},
    'PL': {'Production': [], 'Demand': [], 'Balance_mls': [], 'Balance_uls': []},
    'EE': {'Production': [], 'Demand': [], 'Balance_mls': [], 'Balance_uls': []},
}
countries = data.node.area.unique().tolist()
all_zones = data.node.zone.unique().tolist()
metrics = {
    'Balance_mls': 'Balance_mls',
    'Balance_uls': 'Balance_uls',
    'Production': 'Production',
    'Demand': 'Demand'
}

# Iterate over years and compute sums
for year in energyBalance:
    df = energyBalance[year]['zone_level']  # DataFrame for the year
    for country in countries:
        # Filter zones for the country
        mask = df['Zone'].str.contains(country, na=False)
        for sum_key, df_col in metrics.items():
            # Sum the relevant column for filtered zones
            sumBalance[f'{country}'][sum_key].append(
                df[mask][df_col].astype(float).fillna(0).sum()
            )

# Calculate mean for each country and metric
mean_balance = {
    f'{country}': {
        metric: sum(values) / len(values) if values else 0
        for metric, values in metrics_dict.items()
    }
    for country, metrics_dict in sumBalance.items()
}
mean_balance = pd.DataFrame(mean_balance).T


# #%% Excel ###
# """
# Production, consumption, and price data for specific nodes within a given time period.
#
# Main Features:
# - Handles time using Python's built-in datetime objects.
# - Retrieves simulated production, consumption, and price data from a given SQL file for selected nodes within a specified timeframe.
# - Organizes data and exports it to an Excel file for further analysis.
# """
#
# # === INITIALIZATIONS ===
# START = {"year": 1993, "month": 4, "day": 1, "hour": 0}
# END = {"year": 1993, "month": 9, "day": 30, "hour": 23}
# Nodes = ["SE3_1", "SE3_2", "SE3_3", "SE3_4", "SE3_5", "SE3_6", "SE3_7", "SE3_8","SE3_9", "SE3_10"]
#
# SELECTED_BRANCHES  = None
# # ======================================================================================================================
#
# start_hour, end_hour = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
# production_per_node, gen_idx, gen_type = GetProductionAtSpecificNodes(Nodes, data, database, start_hour, end_hour)
# consumption_per_node = GetConsumptionAtSpecificNodes(Nodes, data, database, start_hour, end_hour)
# nodal_prices_per_node = GetPriceAtSpecificNodes(Nodes, data, database, start_hour, end_hour)
# reservoir_filling_per_node, storage_cap = GetReservoirFillingAtSpecificNodes(Nodes, data, database, start_hour, end_hour)
# flow_data = getFlowDataOnBranches(data, database, [start_hour, end_hour], SELECTED_BRANCHES)
# excel_filename = ExportToExcel(Nodes, production_per_node, consumption_per_node, nodal_prices_per_node, reservoir_filling_per_node, storage_cap, flow_data, START, END, SCENARIO, VERSION, OUTPUT_PATH)


# %%# === Hent ut total produksjon ===

import pandas as pd
from datetime import datetime


def get_zone_production_summary_full_period(data, database, time_Prod, START, END, OUTPUT_PATH):
    '''
    Retrieves production data for the selected nodes over the specified time period,
    returns production by zone and by type for each timestep, converts the results to TWh,
    and merges selected production types into broader categories.

    Parameters:
        SELECTED_NODES (list or str): List of node IDs to include or "ALL" to select all nodes.
        START (dict): Dictionary defining the start time with keys "year", "month", "day", "hour".
        END (dict): Dictionary defining the end time with keys "year", "month", "day", "hour".
        TIMEZONE (str): Timezone name.
        SIM_YEAR_START (datetime): Start of simulation year.
        SIM_YEAR_END (datetime): End of simulation year.
        data (object): Data object containing node information.
        database (object): Database connection or access object for production data.

    Returns:
        zone_summed_df (pd.DataFrame): Production per original production type for each zone, in TWh, with time index.
        zone_summed_merged_df (pd.DataFrame): Production per merged production type for each zone, with total, in TWh, with time index.
    '''
    # Get list of nodes
    Nodes = data.node["id"].dropna().unique().tolist()

    # Get production data
    production_per_node, gen_idx, gen_type = GetProductionAtSpecificNodes(Nodes, data, database, time_Prod[0], time_Prod[1])

    # Create time index
    start_time = pd.Timestamp(datetime(**START))
    end_time = pd.Timestamp(datetime(**END))
    time_index = pd.date_range(start=start_time, end=end_time, freq='h')
    num_timesteps = len(time_index)

    # Initialize dictionary to store time-series data by zone and production type
    zone_production = {}

    # Process production data
    for node, prodtypes in production_per_node.items():
        zone = node.split("_")[0]  # Extract zone from node ID (e.g., 'SE1' from 'SE1_hydro_1')
        if zone not in zone_production:
            zone_production[zone] = {}

        for prodtype, values_list in prodtypes.items():
            # Handle empty or null values
            if not values_list or not values_list[0]:
                values = [0] * num_timesteps
            else:
                values = values_list[0]  # Assume values_list[0] contains the time-series data
                if len(values) != num_timesteps:
                    raise ValueError(
                        f"Production data for node {node}, type {prodtype} has incorrect length: {len(values)} vs {num_timesteps}")

            # Store time-series data
            if prodtype not in zone_production[zone]:
                zone_production[zone][prodtype] = values
            else:
                # Sum production for the same production type in the same zone
                zone_production[zone][prodtype] = [sum(x) for x in zip(zone_production[zone][prodtype], values)]

    # Convert to DataFrame with multi-level columns (zone, prodtype)
    columns = pd.MultiIndex.from_tuples(
        [(zone, prodtype) for zone in zone_production for prodtype in zone_production[zone]],
        names=['Zone', 'Production Type']
    )
    zone_summed_df = pd.DataFrame(
        data=[[zone_production[zone][prodtype][t] for zone in zone_production for prodtype in zone_production[zone]]
              for t in range(num_timesteps)],
        index=time_index,
        columns=columns
    )

    # Convert from MWh to TWh
    # zone_summed_df = zone_summed_df / 1e6

    # Merge production types
    merge_mapping = {
        "Hydro": ["hydro", "ror"],
        "Nuclear": ["nuclear"],
        "Solar": ["solar"],
        "Thermal": ["fossil_gas", "fossil_other", "biomass"],
        "Wind Onshore": ["wind_on"],
        "Wind Offshore": ["wind_off"]
    }

    # Initialize merged DataFrame
    merged_data = {}
    for zone in zone_production:
        for new_type, old_types in merge_mapping.items():
            # Sum the relevant production types for this zone
            valid_types = [t for t in old_types if t in zone_production[zone]]
            if valid_types:
                merged_data[(zone, new_type)] = zone_summed_df[zone][valid_types].sum(axis=1, skipna=True)
            else:
                merged_data[(zone, new_type)] = pd.Series(0, index=time_index)

        # Add total production for the zone
        merged_data[(zone, "Production total")] = zone_summed_df[zone].sum(axis=1, skipna=True)

    # Create merged DataFrame
    merged_columns = pd.MultiIndex.from_tuples(
        [(zone, col) for zone in zone_production for col in ["Production total"] + list(merge_mapping.keys())],
        names=['Zone', 'Production Type']
    )
    zone_summed_merged_df = pd.DataFrame(
        data={col: merged_data[col] for col in merged_columns},
        index=time_index
    )

    zone_summed_df.to_csv(OUTPUT_PATH / f'zone_summed_df_{start_time.year}_{end_time.year}.csv')
    zone_summed_merged_df.to_csv(OUTPUT_PATH / f'zone_summed_merged_df_{start_time.year}_{end_time.year}.csv')

    return zone_summed_df, zone_summed_merged_df

START = {"year": 2007, "month": 1, "day": 1, "hour": 0}
END = {"year": 2007, "month": 12, "day": 31, "hour": 23}
time_Prod = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
# ======================================================================================================================
zone_summed_df, zone_summed_merged_df = get_zone_production_summary_full_period(data, database, time_Prod, START, END, OUTPUT_PATH / 'data_files')

# %%# === Hent ut total produksjon ===
START = {"year": 2007, "month": 1, "day": 1, "hour": 0}
END = {"year": 2007, "month": 12, "day": 31, "hour": 23}
time_period = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)

save_production_to_excel(data, database, time_period, START, END, TIMEZONE, OUTPUT_PATH / 'data_files', VERSION)
