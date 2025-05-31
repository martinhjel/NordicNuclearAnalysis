from functions.work_functions import *
from functions.global_functions import *  # Functions like 'read_grid_data', 'solve_lp' m.m.
from functions.database_functions import  * # Functions like 'getSystemCostFromDB' m.m.
from zoneinfo import ZoneInfo
from powergama.database import Database  # Import Database-Class specifically


# === General Configurations ===
SIM_YEAR_START = 1991           # Start year for the main simulation  (SQL-file)
SIM_YEAR_END = 2020             # End year for the main simulation  (SQL-file)
CASE_YEAR = 2035
SCENARIO = 'BL'
VERSION = 'v41'
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

# Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ['cmr10'],
            "axes.formatter.use_mathtext": True,  # Fix cmr10 warning
            "axes.unicode_minus": False  # Fix minus sign rendering
        })



# %% === PRODUCTION / DEMAND / PRICE / FLOW / RESERVOIR ===

START = {"year": 1991, "month": 1, "day": 1, "hour": 0}
END = {"year": 1991, "month": 12, "day": 31, "hour": 23}
time_period = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
relative_storage = True # True if relative storage in percentage

save_production_to_excel(data, database, time_period, START, END, TIMEZONE,
                         OUTPUT_PATH / 'data_files', VERSION, relative_storage)



# %% === GET SENSIBILITY RANKING FOR SPECIFIC GEN TYPE ===

# === INITIALIZATIONS ===
GEN_TYPE = 'wind_off'
START = {"year": 1991, "month": 1, "day": 1, "hour": 0}
END = {"year": 2020, "month": 12, "day": 31, "hour": 23}
time_period = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
sensitivity_rank = generatorSensitivityRanking(data, database, GEN_TYPE, time_period, weighted=True)


# %% === GET Sensitivity for All Generators and Rank by Type ===

START = {"year": 1991, "month": 1, "day": 1, "hour": 0}
END = {"year": 1991, "month": 12, "day": 31, "hour": 23}
time_period = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
sens = generatorSensitivityRankingALL(data,
                                      database,
                                      time_period,
                                      OUTPUT_PATH_PLOTS,
                                      inflow_weighted=True,
                                      save_fig=True,
                                      include_fliers=True,
                                      area_filter=None)



# %% === ZONAL PRICE MAP ===
zones = ['NO1', 'NO2', 'NO3', 'NO4', 'NO5', 'SE1', 'SE2', 'SE3', 'SE4',
         'DK1', 'DK2', 'FI', 'DE', 'GB', 'NL', 'LT', 'PL', 'EE']
year_range = list(range(SIM_YEAR_START, SIM_YEAR_END + 1))
price_matrix, log = createZonePriceMatrix(data, database, zones, year_range, TIMEZONE, SIM_YEAR_START, SIM_YEAR_END)
# Plot Zonal Price Matrix
colormap = "YlOrRd"
title_map = None # "Average Zonal Price Map"
plot_config = {
    'save_fig': True,  # Save the figure
    'OUTPUT_PATH_PLOTS': OUTPUT_PATH_PLOTS,
    'start': SIM_YEAR_START,
    'end': SIM_YEAR_END,
    'version': VERSION,
    'colormap': colormap,
    'title': title_map,
    'fig_size': (10, 5),  # Figure size in inches
    'dpi': 300,  # Dots per inch for the saved figure
    'cbar_label': "Price [€/MWh]",  # Colorbar label
    'cbar_xpos': 0.02,  # Colorbar x-position padding
    'bbox_inches': 'tight',  # Bounding box for saving the figure,
    'rotation_x': 65,  # Rotation for x-axis labels
    'rotation_y': 0,  # Rotation for y-axis labels
    'ha_x': 'right',  # Horizontal alignment for x-axis labels
    'va_x': 'top',  # Vertical alignment for x-axis labels
    'ha_y': 'right',  # Horizontal alignment for y-axis labels
    'va_y': 'center',  # Vertical alignment for y-axis labels
    'x_label_xShift': 5 / 72,  # X-axis label x-shift
    'x_label_yShift': 0.01,  # X-axis label y-shift

}
plotZonePriceMatrix(price_matrix, plot_config)



# %% === GET INFLOW DEVIATION ===
average_inflow = plot_inflow_deviation(data)

# %% Nordic Grid Map

# === INITIALIZATIONS ===
START = {"year": 1991, "month": 1, "day": 1, "hour": 0}
END = {"year": 2020, "month": 12, "day": 31, "hour": 23}

nordic_grid_map_fromDB(data, database, time_range = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END),
                       OUTPUT_PATH = OUTPUT_PATH / 'maps', version = VERSION, START = START, END = END, exchange_rate_NOK_EUR = 11.38)



# %% === GET ENERGY MIX ===
START = {"year": 1991, "month": 1, "day": 1, "hour": 0}
END = {"year": 1992, "month": 12, "day": 31, "hour": 23}
time_Shed = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
variable_type = "spilled"  # capacity, energy or spilled
dfplot = plotEnergyMix(data=data, database=database, areas=['NO', 'SE', 'FI', 'DK'],
                       timeMaxMin=time_Shed, variable=variable_type).fillna(0)

# %% === GET ENERGY BALANCE ON NODAL AND ZONAL LEVEL ===
# === INITIALIZATIONS ===
YEARS = 30          # Number of years to simulate
# Get energy balance for the specified years
energyBalance, mean_balance = getEnergyBalance(data, database, SIM_YEAR_START, SIM_YEAR_END,
                                               TIMEZONE, OUTPUT_PATH, VERSION, YEARS)




# %% TETS - NY APPROACH BASERT PÅ TIDSSTEG OG IKKE DATO! National-level electricity production and consumption
"""
Retrieves and aggregates electricity production and demand data at the national level.

Based on idealyears over the 30-year climate periode so that each year has same number of hours so it can be comparable to each other.

Overview:

"""
areas = data.node.area.unique().tolist()  # List of areas in the system
areas = areas[0:4]
# === INITIALIZATIONS ===
country = "FI"  # Country code
gen_dict = {}

n_ideal_years = 30
n_timesteps = int(8766.4 * n_ideal_years) # Ved full 30-års simuleringsperiode
# n_timesteps=8760

df_gen, df_prices, total_production, df_gen_per_year = get_production_by_type_ideal_timestep(
        data=data,
        db=database,
        area_OP=country,
        n_timesteps=n_timesteps
    )





# %% PLOT STORAGE FILLING FOR AREAS

# === INITIALIZATIONS ===
START = {"year": 1991, "month": 1, "day": 1, "hour": 0}
END = {"year": 2020, "month": 12, "day": 31, "hour": 23}
areas = ['NO', 'SE', 'FI']  # Areas for plotting
# === PLOT CONFIGURATIONS ===
for area in areas:
    plot_config = {
        'areas': [area],            # When plotting multiple years in one year, recommend to only use one area
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
# === INITIALIZATIONS ===
START = {"year": 1991, "month": 1, "day": 1, "hour": 0}
END = {"year": 2020, "month": 12, "day": 31, "hour": 23}
zones = ['NO1', 'NO2', 'NO3', 'NO4', 'NO5', 'SE1', 'SE2', 'SE3', 'SE4']
# === PLOT CONFIGURATIONS ===
for zone in zones:
    plot_config = {
        'zones': [zone],               # When plotting multiple years in one year, recommend to only use one zone
        'relative': True,               # Relative storage filling, True gives percentage
        "plot_by_year": 3,              # (1) Each year in individual plot, (2) Entire Timeline, (3) Each year show over 1 year timeline.
        "duration_curve": False,        # True: Plot duration curve, or False: Plot storage filling over time
        "save_fig": True,              # True: Save plot as pdf
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





# %% GET FLOW ON SPECIFIC BRANCHES

# === INITIALIZATIONS ===
START = {"year": 1991, "month": 1, "day": 1, "hour": 0}
END = {"year": 1992, "month": 12, "day": 31, "hour": 23}

# === PLOT CONFIGURATIONS ===

plot_config = {
    'areas': ['NO'],            # When plotting multiple years in one year, recommend to only use one area
    "plot_by_year": False,      # Each year in individual plot or all years collected in one plot
    "duration_curve": True,     # True: Plot duration curve, or False: Plot storage filling over time
    "duration_relative": True, # Hours(False) or Percentage(True)
    "save_fig": False,          # True: Save plot as pdf
    "interval": 1,              # Number of months on x-axis. 1 = Step is one month, 12 = Step is 12 months
    "check": False,
    "tex_font": True,
    "fig_size": (6, 3.75),        # Figure size in inches
    "title": None,
    "dpi": 300,  # Dots per inch for the saved figure
    'bbox_inches': 'tight',  # Bounding box for saving the figure,

}

# === CHOOSE BRANCHES TO CHECK ===
SELECTED_BRANCHES  = [['NL','NO2_4'],['NO2_4','DE'], ['NO2_1','GB'], ['DK1_1','NO2_5'], ['DK1_3','GB']] # See branch CSV files for correct connections
# SELECTED_BRANCHES  = [['SE2_3','SE3_1'],['SE3_1','SE2_7'], ['SE3_1','SE2_6'], ['SE3_5','SE2_5'], ['SE2_6','SE3_2']]
# SELECTED_BRANCHES  = [['SE1_2','SE2_2'],['SE1_3','SE2_2'], ['FI_3','SE1_2'], ['FI_3','SE1_3'], ['NO4_1','SE1_1']]
# SELECTED_BRANCHES = [['NL','NO2_4'],['NO2_4','DE'], ['NO2_1','GB'], ['DK1_1','NO2_5']] # NO
# SELECTED_BRANCHES = [['SE4_2','DE'], ['SE3_10','LT'], ['SE4_1','PL'], ['DK2_2','SE4_2'], ['SE3_7','DK1_1'], ['SE3_1','FI_10'], ['SE3_3','FI_10']] # SE
# SELECTED_BRANCHES = [['DK2_2','DE'], ['DK2_hub','DE'], ['DK2_hub','DK2_2'], ['DE','DK1_3']]

# SELECTED_BRANCHES = [['SE4_1','PL'],['SE3_10','LT'], ['SE4_2','DE']]

# === COMPUTE TIMERANGE AND PLOT FLOW ===
time_Lines = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
plot_Flow_fromDB(data, database, DATE_START, time_Lines, OUTPUT_PATH_PLOTS, plot_config, SELECTED_BRANCHES)


# %% PLOT ZONAL PRICES

# === INITIALIZATIONS ===
START = {"year": 1991, "month": 1, "day": 1, "hour": 0}
END = {"year": 1991, "month": 12, "day": 31, "hour": 23}

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






