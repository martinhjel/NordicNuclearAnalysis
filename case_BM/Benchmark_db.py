from functions.more_functions import *
from functions.global_functions import *  # Functions like 'read_grid_data', 'solve_lp' m.m.
from functions.database_functions import  * # Functions like 'getSystemCostFromDB' m.m.


# === General Configurations ===
YEAR_SCENARIO = 2025
YEAR_START = 1991
YEAR_END = 2020
case = 'BM'
version = '52_v20'


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
START_YEAR = 2000
END_YEAR = 2000

time_Map = get_time_steps_for_period(START_YEAR, END_YEAR)

# If one smaller period is wanted, choose time_Map = [start, end] for the given timestep you want
plot_Map(data, database, time_Map, OUTPUT_PATH, version)



# %% GET FLOW ON CHOSEN BRANCHES

# === INITIALIZATIONS ===
START_YEAR = 2000
END_YEAR = 2000

# === PLOT CONFIGURATIONS ===
PLOT_BY_YEAR = False       # Each year in individual plot or all years collected in one plot
DURATION_CURVE = False
DURATION_RELATIVE = True   # Hours(False) or Percentage(True)
SAVE_FIG = False
AXIS_INTERVAL = 1         # Number of months on x-axis. 1 = Step is one month, 12 = Step is 12 months

# === CHOOSE BRANCHES TO CHECK ===
# See branch CSV files for correct connections
SELECTED_BRANCHES  = [['DK1_3','DK1_1'],['FI_3','SE1_2'], ['SE4_2','DE']]

# === COMPUTE TIME RANGE ===
time_Lines = get_time_steps_for_period(START_YEAR, END_YEAR)
correct_date_start_Lines = DATE_START + pd.Timedelta(hours=time_Lines[0])

# === PLOT FLOW ===
plot_Flow_fromDB(db=database, DATE_START=correct_date_start_Lines, time_max_min=time_Lines, grid_data_path=GRID_DATA_PATH,
                 OUTPUT_PATH_PLOTS=OUTPUT_PATH_PLOTS, by_year=PLOT_BY_YEAR, duration_curve=DURATION_CURVE,
                 duration_relative=DURATION_RELATIVE, save_fig=SAVE_FIG, interval=AXIS_INTERVAL, check=False,
                 tex_font=False, chosen_connections=SELECTED_BRANCHES)



# %% Storage Filling

# === INITIALIZATIONS ===
START_YEAR = 2000
END_YEAR = 2000

# === PLOT CONFIGURATIONS ===
plot_config = {
    'areas': ['NO'],            # When plotting multiple years in one year, recommend to only use one area
    'relative': True,           # Relative storage filling, True gives percentage
    "plot_by_year": False,      # Each year in individual plot or all years collected in one plot
    "duration_curve": True,     # True: Plot duration curve, or False: Plot storage filling over time
    "save_fig": False,          # True: Save plot as pdf
    "interval": 12              # Number of months on x-axis. 1 = Step is one month, 12 = Step is 12 months
}


time_SF = get_time_steps_for_period(START_YEAR, END_YEAR)
plot_SF_Areas_FromDB(data, database, time_SF, OUTPUT_PATH_PLOTS, DATE_START, plot_config)

# %%
# Her trengs det fortsatt litt jobb med scaleringen av selve plottet, men det er ikke krise enda.


# === INITIALIZATIONS ===
START_YEAR = 2000
END_YEAR = 2000


# === PLOT CONFIGURATIONS ===
plot_config = {
    'zones': ['NO1', 'NO2', 'NO3', 'NO4', 'NO5'],   # When plotting multiple years in one year, recommend to only use one zone
    'relative': True,                               # Relative storage filling, True gives percentage
    "plot_by_year": False,                           # Each year in individual plot or all years collected in one plot
    "duration_curve": False,                         # True: Plot duration curve, or False: Plot storage filling over time
    "save_fig": False,                              # True: Save plot as pdf
    "interval": 1                                   # Number of months on x-axis. 1 = Step is one month, 12 = Step is 12 months
}

# If you want to go in and change title, follow the function from here to its source location and change it there.
# Remember that you then have to reset the console run
time_SF = get_time_steps_for_period(START_YEAR, END_YEAR)
plot_SF_Zones_FromDB(data, database, time_SF, OUTPUT_PATH_PLOTS, DATE_START, plot_config)




# %% Plot nodal prices Norway


# === INITIALIZATIONS ===
START_YEAR = 2000
END_YEAR = 2000

# === PLOT CONFIGURATIONS ===
plot_config = {
    'zone': 'NO3',                                     # When plotting multiple years in one year, recommend to only use one zone
    'plot_all_nodes': False,                           # Relative storage filling, True gives percentage (HØRER DENNE TIL ET ANNET PLOT?)
    "plot_by_year": False,                             # Each year in individual plot or all years collected in one plot
    "duration_curve": False,                           # True: Plot duration curve, or False: Plot storage filling over time
    "save_fig": False,                                 # True: Save plot as pdf
    "interval": 1                                      # Number of months on x-axis. 1 = Step is one month, 12 = Step is 12 months
}


time_NP = get_time_steps_for_period(START_YEAR, END_YEAR)
calcPlot_NP_FromDB(data, database, time_NP, OUTPUT_PATH_PLOTS, DATE_START, plot_config)


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

time_HRI = get_time_steps_for_period(START_YEAR, END_YEAR)
calcPlot_HRI_FromDB(data, database, time_HRI, OUTPUT_PATH_PLOTS, DATE_START, plot_config)


# %% Plot nodal prices, demand and hydro production

# TODO Fix denne

def calcPlot_PLP_FromDB(data: GridData, database: Database, time_max_min, OUTPUT_PATH_PLOTS, DATE_START):
    area_OP = 'NO'
    title = "Avg. Area Price, Demand and Hydro Production in NO"
    plot_full_timeline = True
    save_fig_PDP = False
    interval = 1
    box_in_frame = True
    resample = True

    time_PLP = time_max_min# get_time_steps_for_period(2000, 2001)
    correct_date_start_PLP = DATE_START + pd.Timedelta(hours=time_PLP[0])
    correct_date_end_PLP = DATE_START + pd.Timedelta(hours=time_PLP[-1])


    df_plp, df_plp_resampled = calc_PLP_FromDB(data, database, area_OP, correct_date_start_PLP, time_PLP)
    plot_hydro_prod_demand_price(df_plp=df_plp,
                                 df_plp_resampled=df_plp_resampled,
                                 resample=resample,
                                 DATE_START=correct_date_start_PLP,
                                 DATE_END=correct_date_end_PLP,
                                 interval=interval,
                                 TITLE=title,
                                 save_fig=save_fig_PDP,
                                 plot_full_timeline=plot_full_timeline,
                                 box_in_frame=box_in_frame,
                                 OUTPUT_PATH_PLOTS=OUTPUT_PATH_PLOTS,
                                 tex_font=False)

calcPlot_PLP_FromDB(data, database, time_max_min, OUTPUT_PATH_PLOTS, DATE_START)


# %% Load, generation by type in AREA
# TODO fix denne

def calcPlot_LG_FromDB(data: GridData, database: Database, time_max_min, OUTPUT_PATH_PLOTS, DATE_START, area_OP):
    title_gen = f'Production, Consumption and Price in {area_OP}'
    interval = 1
    figsize = (10, 6)

    time_LGT = time_max_min# get_time_steps_for_period(2000, 2003)
    correct_date_start_LGT = DATE_START + pd.Timedelta(hours=time_LGT[0])
    correct_date_end_LGT = DATE_START + pd.Timedelta(hours=time_LGT[-1])


    df_gen_resampled, df_prices_resampled, total_production = get_production_by_type_FromDB(data,
                                                                                            database,
                                                                                            area_OP,
                                                                                            time_LGT,
                                                                                            correct_date_start_LGT)
    plot_full_timeline = True
    plot_duration_curve = False
    save_plot_LG = True
    box_in_frame_LG = False
    plot_production(df_gen_resampled=df_gen_resampled,
                    df_prices_resampled=df_prices_resampled,
                    DATE_START=correct_date_start_LGT,
                    DATE_END=correct_date_end_LGT,
                    interval=interval,
                    fig_size=figsize,
                    TITLE=title_gen,
                    OUTPUT_PATH_PLOTS=OUTPUT_PATH_PLOTS,
                    plot_full_timeline=plot_full_timeline,
                    plot_duration_curve=plot_duration_curve,
                    save_fig=save_plot_LG,
                    box_in_frame=box_in_frame_LG,
                    tex_font=False)

    return df_gen_resampled, df_prices_resampled, total_production


area='DK'
df_gen_re, df_prices, tot_prod = calcPlot_LG_FromDB(data, database, time_max_min, OUTPUT_PATH_PLOTS, DATE_START, area_OP=area)
print(f"Total production in {area}: {tot_prod:.2f} MWh")


# %% Production data for area or zone

""" Initialize data for Production"""
# La area eller zone være None om de ikke skal brukes.

# === INITIALIZATIONS ===
START_YEAR = 2000
END_YEAR = 2000

area = None
zone = 'NO1'

# Juster area for å se på sonene, og zone for å se på nodene i sonen
time_Prod = get_time_steps_for_period(START_YEAR, END_YEAR)
correct_date_start_Prod = DATE_START + pd.Timedelta(hours=time_Prod[0])
if area is not None:
    zones_in_area_prod = getProductionZonesInArea(data, database, area, time_Prod, correct_date_start_Prod, week=True)
if zone is not None:
    nodes_in_zone_prod = getProductionNodesInZone(data, database, zone, time_Prod, correct_date_start_Prod, week=True)

