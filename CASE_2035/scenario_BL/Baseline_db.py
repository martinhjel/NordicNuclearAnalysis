from functions.work_functions import *
from functions.global_functions import *  # Functions like 'read_grid_data', 'solve_lp' m.m.
from functions.database_functions import  * # Functions like 'getSystemCostFromDB' m.m.
from zoneinfo import ZoneInfo
from powergama.database import Database  # Import Database-Class specifically
import pandas as pd


# === General Configurations ===
SIM_YEAR_START = 1991           # Start year for the main simulation  (SQL-file)
SIM_YEAR_END = 2020             # End year for the main simulation  (SQL-file)
CASE_YEAR = 2035
SCENARIO = 'BL'
VERSION = 'v9'
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



# %% Nordic Grid Map

# === INITIALIZATIONS ===
START = {"year": 2012, "month": 1, "day": 1, "hour": 0}
END = {"year": 2012, "month": 12, "day": 31, "hour": 23}
nordic_grid_map_fromDB(data, database, time_range = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END),
                       OUTPUT_PATH = OUTPUT_PATH, version = VERSION, START = START, END = END, exchange_rate_NOK_EUR = 11.38)





# %% === ZONAL PRICE MAP ===
# TODO: legg til mulighet for å ha øre/kwh
zones = ['NO1', 'NO2', 'NO3', 'NO4', 'NO5', 'SE1', 'SE2', 'SE3', 'SE4',
         'DK1', 'DK2', 'FI', 'DE', 'GB', 'NL', 'LT', 'PL', 'EE']
START = {"year": 1991, "month": 1, "day": 1, "hour": 0}
END = {"year": 2020, "month": 12, "day": 31, "hour": 23}
year_range = list(range(SIM_YEAR_START, SIM_YEAR_END + 1))
price_matrix, log = createZonePriceMatrix(data, database, zones, year_range, TIMEZONE, SIM_YEAR_START, SIM_YEAR_END)

# %% Plot Zonal Price Matrix
plotZonePriceMatrix(price_matrix, save_fig=True, OUTPUT_PATH_PLOTS=OUTPUT_PATH_PLOTS, start=START, end=END, version=VERSION)



# %% Check Total Consumption for a given period.
# Demand Response
# === INITIALIZATIONS ===
START = {"year": 1993, "month": 1, "day": 1, "hour": 0}
END = {"year": 1993, "month": 12, "day": 31, "hour": 23}
area = 'NO'

time_Demand = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
demandTotal = getDemandPerAreaFromDB(data, database, area=area, timeMaxMin=time_Demand)
print(sum(demandTotal['sum']))


# %% === Get Production Data ===
# === INITIALIZATIONS ===
START = {"year": 1991, "month": 1, "day": 1, "hour": 0}
END = {"year": 1991, "month": 12, "day": 31, "hour": 23}
area = 'SE'

time_Prod = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
total_Production = getProductionPerAreaFromDB(data, database, time_Prod, area)
print(total_Production)



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
    "interval": 1,               # Number of months on x-axis. 1 = Step is one month, 12 = Step is 12 months
}

# === COMPUTE TIMERANGE AND PLOT FLOW ===
time_SF = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
plot_SF_Areas_FromDB(data, database, time_SF, OUTPUT_PATH_PLOTS, DATE_START, plot_config)

# %% PLOT STORAGE FILLING ZONES
# Todo: Trengs det fortsatt litt jobb med scaleringen av selve plottet, men det er ikke krise enda.
# Todo: Må OGSÅ ha mulighet til å plotte storage filling ned på node nivå.

# === INITIALIZATIONS ===
START = {"year": 1991, "month": 1, "day": 1, "hour": 0}
END = {"year": 1995, "month": 12, "day": 31, "hour": 23}

# === PLOT CONFIGURATIONS ===
plot_config = {
    'zones': ['NO1'],                     # When plotting multiple years in one year, recommend to only use one zone
    'relative': True,                            # Relative storage filling, True gives percentage
    "plot_by_year": 3,                           # (1) Each year in individual plot, (2) Entire Timeline, (3) Each year show over 1 year timeline.
    "duration_curve": False,                     # True: Plot duration curve, or False: Plot storage filling over time
    "save_fig": False,                           # True: Save plot as pdf
    "interval": 1                                # Number of months on x-axis. 1 = Step is one month, 12 = Step is 12 months
}

# If you want to go in and change title, follow the function from here to its source location and change it there.
# Remember that you then have to reset the console run
# === COMPUTE TIMERANGE AND PLOT FLOW ===
time_SF = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
plot_SF_Zones_FromDB(data, database, time_SF, OUTPUT_PATH_PLOTS, DATE_START, plot_config)




# %% GET FLOW ON CHOSEN BRANCHES

# === INITIALIZATIONS ===
START = {"year": 2012, "month": 1, "day": 1, "hour": 0}
END = {"year": 2012, "month": 12, "day": 31, "hour": 23}

# === PLOT CONFIGURATIONS ===

plot_config = {
    'areas': ['SE'],            # When plotting multiple years in one year, recommend to only use one area
    "plot_by_year": False,      # Each year in individual plot or all years collected in one plot
    "duration_curve": True,     # True: Plot duration curve, or False: Plot storage filling over time
    "duration_relative": False, # Hours(False) or Percentage(True)
    "save_fig": False,          # True: Save plot as pdf
    "interval": 1,              # Number of months on x-axis. 1 = Step is one month, 12 = Step is 12 months
    "check": False,
    "tex_font": False
}

# === CHOOSE BRANCHES TO CHECK ===
# SELECTED_BRANCHES  = [['NL','NO2_4'],['NO2_4','DE'], ['NO2_1','GB'], ['DK1_1','NO2_5']] # See branch CSV files for correct connections
# SELECTED_BRANCHES  = [['SE2_3','SE3_1'],['SE3_1','SE2_7'], ['SE3_1','SE2_6'], ['SE3_5','SE2_5'], ['SE2_6','SE3_2']]
SELECTED_BRANCHES  = [['SE1_2','SE2_2'],['SE1_3','SE2_2'], ['FI_3','SE1_2'], ['FI_3','SE1_3'], ['NO4_1','SE1_1']]



# === COMPUTE TIMERANGE AND PLOT FLOW ===
time_Lines = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
plot_Flow_fromDB(data, database, DATE_START, time_Lines, OUTPUT_PATH_PLOTS, plot_config, SELECTED_BRANCHES)


# %% PLOT ZONAL PRICES

# === INITIALIZATIONS ===
START = {"year": 2012, "month": 1, "day": 1, "hour": 0}
END = {"year": 2012, "month": 12, "day": 31, "hour": 23}

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
START = {"year": 2012, "month": 1, "day": 1, "hour": 0}
END = {"year": 2012, "month": 12, "day": 31, "hour": 0}

# === PLOT CONFIGURATIONS ===
plot_config = {
    'area': 'NO',
    'title': 'Production, Consumption and Price in SE',
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
tot_prod = df_gen_re.drop('Load', axis=1).sum(axis=1).sum()
print(f"Total production in {plot_config['area']}: {tot_prod:.2f} MWh")


# %%

# === INITIALIZATIONS ===
START = {"year": 2010, "month": 1, "day": 1, "hour": 0}
END = {"year": 2010, "month": 12, "day": 31, "hour": 23}
area = None
zone = 'SE1'

# Juster area for å se på sonene, og zone for å se på nodene i sonen
# === COMPUTE TIMERANGE AND PLOT FLOW ===
time_Prod = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
correct_date_start_Prod = DATE_START + pd.Timedelta(hours=time_Prod[0])
if area is not None:
    zones_in_area_prod = getProductionZonesInArea(data, database, area, time_Prod, correct_date_start_Prod, week=True)
if zone is not None:
    nodes_in_zone_prod = getProductionNodesInZone(data, database, zone, time_Prod, correct_date_start_Prod, week=True)


energyBalance = nodes_in_zone_prod.sum(axis=0)
