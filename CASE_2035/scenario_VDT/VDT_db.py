from functions.work_functions import *
from functions.global_functions import *  # Functions like 'read_grid_data', 'solve_lp' m.m.
from functions.database_functions import  * # Functions like 'getSystemCostFromDB' m.m.
from zoneinfo import ZoneInfo
from powergama.database import Database  # Import Database-Class specifically
import pandas as pd


# === General Configurations ===
SIM_YEAR_START = 1991           # Start year for the main simulation  (SQL-file)
SIM_YEAR_END = 2000             # End year for the main simulation  (SQL-file)
CASE_YEAR = 2035
SCENARIO = 'VDT'
VERSION = 'v1'
TIMEZONE = ZoneInfo("UTC")  # Definerer UTC tidssone

####  PASS PÅ HARD KODING I SQL FIL

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
    BASE_DIR = BASE_DIR / f'CASE_{CASE_YEAR}' / f'scenario_{SCENARIO}'

# === File Paths ===
SQL_FILE = BASE_DIR / f"powergama_{SCENARIO}_{VERSION}_{SIM_YEAR_START}_{SIM_YEAR_END}.sqlite"
DATA_PATH = BASE_DIR / 'data'
GRID_DATA_PATH = DATA_PATH / 'system'
OUTPUT_PATH = BASE_DIR / 'results'
OUTPUT_PATH_PLOTS = BASE_DIR / 'results' / 'plots'

# === Initialize Database and Grid Data ===
data, time_max_min = setup_grid(VERSION, DATE_START, DATE_END, DATA_PATH, SCENARIO)
database = Database(SQL_FILE)


# %% === ZONAL PRICE MAP ===

# TODO: legg til mulighet for å ha øre/kwh
zones = ['NO1', 'NO2', 'NO3', 'NO4', 'NO5', 'SE1', 'SE2', 'SE3', 'SE4',
         'DK1', 'DK2', 'FI', 'DE', 'GB', 'NL', 'LT', 'PL', 'EE']
year_range = list(range(SIM_YEAR_START, SIM_YEAR_END + 1))
price_matrix, log = createZonePriceMatrix(data, database, zones, year_range, TIMEZONE, SIM_YEAR_START, SIM_YEAR_END)
# Plot Zonal Price Matrix
plotZonePriceMatrix(price_matrix, save_fig=True, OUTPUT_PATH_PLOTS=OUTPUT_PATH_PLOTS, start=SIM_YEAR_START, end=SIM_YEAR_END, version=VERSION)


# %%
START = {"year": 1991, "month": 1, "day": 1, "hour": 0}
END = {"year": 1995, "month": 12, "day": 31, "hour": 23}
time_Shed = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
dfplot = plotEnergyMix(data=data, database=database, areas=['NO', 'SE', 'FI', 'DK'], timeMaxMin=time_Shed, variable="capacity")

# %% === GET ENERGY BALANCE ON NODAL AND ZONAL LEVEL ===
energyBalance = {}

for i in range(0, 10):
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
    'NO': {'Production': [], 'Demand': [], 'Balance': []},
    'SE': {'Production': [], 'Demand': [], 'Balance': []},
    'FI': {'Production': [], 'Demand': [], 'Balance': []},
    'DK': {'Production': [], 'Demand': [], 'Balance': []},
    'GB': {'Production': [], 'Demand': [], 'Balance': []},
    'DE': {'Production': [], 'Demand': [], 'Balance': []},
    'NL': {'Production': [], 'Demand': [], 'Balance': []},
    'LT': {'Production': [], 'Demand': [], 'Balance': []},
    'PL': {'Production': [], 'Demand': [], 'Balance': []},
    'EE': {'Production': [], 'Demand': [], 'Balance': []},
}
countries = data.node.area.unique().tolist()
all_zones = data.node.zone.unique().tolist()
metrics = {
    'Balance': 'Balance',
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

# %%
START = {"year": 1991, "month": 1, "day": 1, "hour": 0}
END = {"year": 1991, "month": 12, "day": 31, "hour": 23}
time_EB = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
all_nodes = data.node.id.tolist()
production, gen_idx, gen_type = GetProductionAtSpecificNodes(all_nodes, data, database, time_EB[0], time_EB[1])



# %% TETS - NY APPROACH BASERT PÅ TIDSSTEG OG IKKE DATO! National-level electricity production and consumption
"""
Retrieves and aggregates electricity production and demand data at the national level.

Based on idealyears over the 30-year climate periode so that each year has same number of hours so it can be comparable to each other.


Overview:

"""

# === INITIALIZATIONS ===
country = "PL"  # Country code

n_ideal_years = 3
n_timesteps = int(8766.4 * n_ideal_years) # Ved full 30-års simuleringsperiode
# n_timesteps=8760

df_gen, df_prices, total_production, df_gen_per_year = get_production_by_type_ideal_timestep(
    data=data,
    db=database,
    area_OP=country,
    n_timesteps=n_timesteps
)


# %% === Get production by type aggregate by zone ===
'''
Aggregates electricity production by zone and production type over a specified time period. 
Returns two DataFrames: (1) detailed production per type, and (2) production merged into broader categories, in TWh.

Input:
- SELECTED_NODES = List of node IDs (e.g., ["NO1_1", "NO1_2", "NO1_3"]) 
  or the string "ALL" to include all nodes in the system.
'''
# === INITIALIZATIONS ===
START = {"year": 1993, "month": 1, "day": 1, "hour": 0}
END = {"year": 1993, "month": 12, "day": 31, "hour": 23}
SELECTED_NODES = "ALL"
# SELECTED_NODES = ["SE4_1", "SE4_2", "SE3_1", "SE2_1", "SE2_2"]
# ======================================================================================================================
zone_summed_df, zone_summed_merged_df = get_zone_production_summary(SELECTED_NODES, START, END, TIMEZONE, SIM_YEAR_START, SIM_YEAR_END, data, database)



# %% === CHECK SPILLED VS PRODUCED ===
START = {"year": 2020, "month": 1, "day": 1, "hour": 0}
END = {"year": 2020, "month": 12, "day": 31, "hour": 23}
time_EB = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
gen_idx = [385]
sum_spilled, sum_produced = checkSpilled_vs_ProducedAtGen(database, gen_idx, time_EB)


# %% === GET IMPORTS/EXPORTS FOR EACH ZONE ===

START = {"year": 1991, "month": 1, "day": 1, "hour": 0}
END = {"year": 1994, "month": 12, "day": 31, "hour": 23}
time_EB = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
flow_data = getFlowDataOnALLBranches(data, database, time_EB)

zone_imports, zone_exports = getZoneImportExports(data, flow_data)
# Example: Print results
print("Zone Imports (importer, exporter): Total Import [MWh]")
for (importer, exporter), total in zone_imports.items():
    print(f"{importer} importing from {exporter}: {total:.2f} MWh")




# %% Nordic Grid Map

# === INITIALIZATIONS ===
START = {"year": 1991, "month": 1, "day": 1, "hour": 0}
END = {"year": 2020, "month": 12, "day": 31, "hour": 23}

nordic_grid_map_fromDB(data, database, time_range = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END),
                       OUTPUT_PATH = OUTPUT_PATH / 'maps', version = VERSION, START = START, END = END, exchange_rate_NOK_EUR = 11.38)



# %% Check Total Consumption for a given period.
# Demand Response
# === INITIALIZATIONS ===
START = {"year": 2020, "month": 1, "day": 1, "hour": 0}
END = {"year": 2020, "month": 12, "day": 31, "hour": 23}
area = 'NO'

time_Demand = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
demandTotal = getDemandPerAreaFromDB(data, database, area=area, timeMaxMin=time_Demand)
print(sum(demandTotal['sum']))


# %% === Get Production Data ===
# === INITIALIZATIONS ===
START = {"year": 2020, "month": 1, "day": 1, "hour": 0}
END = {"year": 2020, "month": 12, "day": 31, "hour": 23}
area = 'PL'

time_Prod = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
total_Production = getProductionPerAreaFromDB(data, database, time_Prod, area)
print(total_Production)



# %% PLOT STORAGE FILLING FOR AREAS

# === INITIALIZATIONS ===
START = {"year": 1991, "month": 1, "day": 1, "hour": 0}
END = {"year": 2020, "month": 12, "day": 31, "hour": 23}

# For at TEX-fonts skal kjøre, må du kjøre en test run først for å initialisere tex-fontene.
# === PLOT CONFIGURATIONS ===
plot_config = {
    'areas': ['FI'],            # When plotting multiple years in one year, recommend to only use one area
    'relative': True,           # Relative storage filling, True gives percentage
    "plot_by_year": True,       # True: One curve for each year in same plot, or False:all years collected in one plot over the whole simulation period
    "duration_curve": False,    # True: Plot duration curve, or False: Plot storage filling over time
    "save_fig": False,           # True: Save plot as pdf
    "interval": 1,              # Number of months on x-axis. 1 = Step is one month, 12 = Step is 12 months
    'empty_threshold': 1e-6,    # If relative (True), empty_threshold is in percentage, if not, it is in MWh
    'include_legend': False,     # Include legend in the plot
    'fig_size': (12, 6),        # Figure size in inches
    'tex_font': True,          # Keep false unless tex packages are installed.
                                # Kan hende må kjøres et par ganger for å få det til å funke med texfont.
}

# === COMPUTE TIMERANGE AND PLOT FLOW ===
time_SF = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
OUTPUT_PATH_PLOTS_RESERVOIR = OUTPUT_PATH_PLOTS / 'reservoir'
plot_SF_Areas_FromDB(data, database, time_SF, OUTPUT_PATH_PLOTS_RESERVOIR, DATE_START, plot_config, START, END)

# %% PLOT STORAGE FILLING ZONES
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
    'fig_size': (12, 6),            # Figure size in inches
    'tex_font': True,              # Keep false unless tex packages are installed
}

# If you want to go in and change title, follow the function from here to its source location and change it there.
# Remember that you then have to reset the console run
# === COMPUTE TIMERANGE AND PLOT FLOW ===
time_SF = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
OUTPUT_PATH_PLOTS_RESERVOIR = OUTPUT_PATH_PLOTS / 'reservoir'
plot_SF_Zones_FromDB(data, database, time_SF, OUTPUT_PATH_PLOTS_RESERVOIR, DATE_START, plot_config, START, END)





# %% GET FLOW ON CHOSEN BRANCHES

# === INITIALIZATIONS ===
START = {"year": 1991, "month": 1, "day": 1, "hour": 0}
END = {"year": 2020, "month": 12, "day": 31, "hour": 23}

# === PLOT CONFIGURATIONS ===

plot_config = {
    'areas': ['SE'],            # When plotting multiple years in one year, recommend to only use one area
    "plot_by_year": False,      # Each year in individual plot or all years collected in one plot
    "duration_curve": True,     # True: Plot duration curve, or False: Plot storage filling over time
    "duration_relative": True, # Hours(False) or Percentage(True)
    "save_fig": False,          # True: Save plot as pdf
    "interval": 1,              # Number of months on x-axis. 1 = Step is one month, 12 = Step is 12 months
    "check": False,
    "tex_font": True
}

# === CHOOSE BRANCHES TO CHECK ===
# SELECTED_BRANCHES  = [['NL','NO2_4'],['NO2_4','DE'], ['NO2_1','GB'], ['DK1_1','NO2_5']] # See branch CSV files for correct connections
# SELECTED_BRANCHES  = [['SE2_3','SE3_1'],['SE3_1','SE2_7'], ['SE3_1','SE2_6'], ['SE3_5','SE2_5'], ['SE2_6','SE3_2']]
# SELECTED_BRANCHES  = [['SE1_2','SE2_2'],['SE1_3','SE2_2'], ['FI_3','SE1_2'], ['FI_3','SE1_3'], ['NO4_1','SE1_1']]
SELECTED_BRANCHES = [['NL','NO2_4'],['NO2_4','DE'], ['NO2_1','GB'], ['DK1_1','NO2_5']] # NO
# SELECTED_BRANCHES = [['SE4_2','DE'], ['SE3_10','LT'], ['SE4_1','PL'], ['DK2_2','SE4_2'], ['SE3_7','DK1_1'], ['SE3_1','FI_10'], ['SE3_3','FI_10']] # SE
# SELECTED_BRANCHES = [['DK2_2','DE'], ['DK2_hub','DE'], ['DE','DK1_3']]

# SELECTED_BRANCHES = [['SE4_1','PL'],['SE3_10','LT'], ['SE4_2','DE']]

# === COMPUTE TIMERANGE AND PLOT FLOW ===
time_Lines = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
plot_Flow_fromDB(data, database, DATE_START, time_Lines, OUTPUT_PATH_PLOTS, plot_config, SELECTED_BRANCHES)


# %% PLOT ZONAL PRICES

# === INITIALIZATIONS ===
START = {"year": 1994, "month": 1, "day": 1, "hour": 0}
END = {"year": 1994, "month": 12, "day": 31, "hour": 23}

# === PLOT CONFIGURATIONS ===
plot_config = {
    'zones': ['NO2'],                       # Zones for plotting
    "plot_by_year": True,                   # (True)Each year in individual plot or (False) all years collected in one plot
    "duration_curve": False,                # True: Plot duration curve, or False: Plot storage filling over time
    "save_fig": False,                      # True: Save plot as pdf
    "interval": 1,                          # Number of months on x-axis. 1 = Step is one month, 12 = Step is 12 months
    "tex_font": False                       # Keep false
}

# === COMPUTE TIMERANGE AND PLOT FLOW ===
time_ZP = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
calcPlot_ZonalPrices_FromDB(data, database, time_ZP, OUTPUT_PATH_PLOTS, DATE_START, plot_config)


# %% Load, generation by type in AREA

# Siden det er ukes aggregert, bør man trekke fra 23 timer for vanlig år, og 1d23t for skuddår
# for et godt plot.
# === INITIALIZATIONS ===
START = {"year": 1994, "month": 1, "day": 1, "hour": 0}
END = {"year": 1994, "month": 12, "day": 31, "hour": 0}

# === PLOT CONFIGURATIONS ===
plot_config = {
    'area': 'GB',
    'title': 'Production, Consumption and Price in GB',
    "fig_size": (15, 10),
    "plot_full_timeline": True,
    "duration_curve": False,
    "box_in_frame": False,
    "save_fig": True,                      # True: Save plot as pdf
    "interval": 1                           # Number of months on x-axis. 1 = Step is one month, 12 = Step is 12 months
}


# === COMPUTE TIMERANGE AND PLOT FLOW ===
time_LGT = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
df_gen_re, df_prices, tot_prod = calcPlot_LG_FromDB(data, database, time_LGT, OUTPUT_PATH_PLOTS, DATE_START, plot_config)
tot_prod = df_gen_re.drop('Load', axis=1).sum(axis=1).sum()
print(f"Total production in {plot_config['area']}: {tot_prod:.2f} MWh")



#%% Excel ###
"""
Production, consumption, and price data for specific nodes within a given time period.

Main Features:
- Handles time using Python's built-in datetime objects.
- Retrieves simulated production, consumption, and price data from a given SQL file for selected nodes within a specified timeframe.
- Organizes data and exports it to an Excel file for further analysis.
"""

# === INITIALIZATIONS ===
START = {"year": 1992, "month": 4, "day": 14, "hour": 0}
END = {"year": 1992, "month": 5, "day": 31, "hour": 23}
Nodes = ["DK1_2", "DK1_3", "SE3_3", "SE3_9"]
SELECTED_BRANCHES  = [['SE3_1','FI_10'], ['SE3_3', 'FI_10']]
# ======================================================================================================================

start_hour, end_hour = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
production_per_node, gen_idx, gen_type = GetProductionAtSpecificNodes(Nodes, data, database, start_hour, end_hour)


consumption_per_node = GetConsumptionAtSpecificNodes(Nodes, data, database, start_hour, end_hour)
nodal_prices_per_node = GetPriceAtSpecificNodes(Nodes, data, database, start_hour, end_hour)
reservoir_filling_per_node, storage_cap = GetReservoirFillingAtSpecificNodes(Nodes, data, database, start_hour, end_hour)
flow_data = getFlowDataOnBranches(data, database, [start_hour, end_hour], SELECTED_BRANCHES)
excel_filename = ExportToExcel(Nodes, production_per_node, consumption_per_node, nodal_prices_per_node, reservoir_filling_per_node, storage_cap, flow_data, START, END, SCENARIO, VERSION, OUTPUT_PATH)


