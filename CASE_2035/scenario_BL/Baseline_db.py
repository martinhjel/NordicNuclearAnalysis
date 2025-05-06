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
VERSION = 'v31'
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




# %%

def plotInflowInArea(data: GridData, area, date_start, date_end):

    genTypeIdx = data.getGeneratorsPerAreaAndType()[area]['hydro']
    if area in ['SE', 'FI']:
        genTypeIdx.extend(data.getGeneratorsPerAreaAndType()[area].get('ror', []))
    # Inflow
    inflowFactor = data.generator["inflow_fac"][
        genTypeIdx[0]]  # Assuming all generators have the same inflow factor
    inflowProfile = data.generator["inflow_ref"][
        genTypeIdx]  # All generators have different inflow profiles, zone-based
    inflow_df = pd.DataFrame()

    for gen in genTypeIdx:
        prodCap = data.generator['pmax'][gen]  # MW
        inflow_value = data.profiles[inflowProfile[gen]]
        inflow_df[gen] = [i * prodCap * inflowFactor for i in inflow_value]
    inflow = inflow_df.sum(axis=1)
    # Slice the inflow series using index-based selection
    # inflow = inflow.iloc[time_max_min[0]:time_max_min[1]]  # +1 to include the max index
    print(f"Total inflow: {sum(inflow) / 1e6:.2f} TWh")

    # Plot inflow
    # Plot setup
    import matplotlib
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.serif'] = ['cmr10']  # Computer Modern Roman
    matplotlib.rcParams['axes.formatter.use_mathtext'] = True  # Fix cmr10 warning
    matplotlib.rcParams['axes.unicode_minus'] = False  # Fix minus sign rendering
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(inflow.index, inflow.values, label='Inflow', color='blue')
    plt.title('Inflow over Time (NO)')
    plt.xlabel('Year')
    plt.ylabel('Inflow (MWh/h)')
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.show()
    return inflow


def plotInflowInZone(data: GridData, zone, date_start, date_end):

    genTypeIdx = data.getGeneratorsPerZoneAndType()[zone]['hydro']
    if zone in ['SE1','SE2','SE3','SE4','FI']:
        genTypeIdx.extend(data.getGeneratorsPerZoneAndType()[zone].get('ror', []))
    # Inflow
    inflowFactor = data.generator["inflow_fac"][
        genTypeIdx[0]]  # Assuming all generators have the same inflow factor
    inflowProfile = data.generator["inflow_ref"][
        genTypeIdx]  # All generators have different inflow profiles, zone-based
    inflow_df = pd.DataFrame()

    for gen in genTypeIdx:
        prodCap = data.generator['pmax'][gen]  # MW
        inflow_value = data.profiles[inflowProfile[gen]]
        inflow_df[gen] = [i * prodCap * inflowFactor for i in inflow_value]
    inflow = inflow_df.sum(axis=1)
    # Slice the inflow series using index-based selection
    # inflow = inflow.iloc[time_max_min[0]:time_max_min[1]]  # +1 to include the max index
    print(f"Total inflow: {sum(inflow) / 1e6:.2f} TWh")

    # Ensure the index is datetime
    inflow.index = pd.date_range(
        start=f"{date_start['year']}-{date_start['month']}-{date_start['day']} {date_start['hour']}:00",
        end=f"{date_end['year']}-{date_end['month']}-{date_end['day']} {date_end['hour']}:00",
        freq='h')

    # Resample to yearly data (sum or mean, depending on your preference)
    # inflow_yearly = inflow.resample('Y').sum()  # Use 'Y' for year-end frequency, sums hourly inflow for each year

    # Plot inflow
    # Plot setup
    import matplotlib
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.serif'] = ['cmr10']  # Computer Modern Roman
    matplotlib.rcParams['axes.formatter.use_mathtext'] = True  # Fix cmr10 warning
    matplotlib.rcParams['axes.unicode_minus'] = False  # Fix minus sign rendering
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(inflow.index.year, inflow.values, label='Inflow', color='blue')
    plt.title('Inflow over Time (NO)')
    plt.xlabel('hours')
    plt.ylabel('Inflow (MWh/h)')
    plt.legend()
    plt.tight_layout()
    plt.grid()

    # Set x-axis to show integer years
    ax.set_xticks(range(date_start['year'], date_end['year'] + 1, 1))  # Show every year
    ax.set_xticklabels(range(date_start['year'], date_end['year'] + 1, 1), rotation=45)

    plt.show()
    return inflow

START = {"year": 1991, "month": 1, "day": 1, "hour": 0}
END = {"year": 2020, "month": 12, "day": 31, "hour": 23}
time_EB = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
zones = 'NO1'
inflow = plotInflowInZone(data, zones, OUTPUT_PATH_PLOTS, START, END)

# %%
def plotInflowValueBox(data: GridData, zone, OUTPUT_PATH_PLOTS, date_start, date_end):
    # Extract inflow value and time index for the specific zone
    inflow_value = data.profiles[f'inflow_{zone}']  # e.g., 'inflow_NO1'
    time_index = data.profiles['time']

    # inflow_series = pd.Series(inflow_value, index=time_index)

    # Debugging: Verify data
    # print(f"Length of inflow_{zone}: {len(inflow_value)}")
    # print(f"Length of time_index: {len(time_index)}")
    # print(f"Sample inflow values (first 5): {inflow_value[:5]}")
    # print(f"Time index range: {time_index.min()} to {time_index.max()}")

    # Check for non-numeric or NaN values in inflow_value
    inflow_array = np.array(inflow_value, dtype=object)  # Avoid premature type coercion
    non_numeric = [x for x in inflow_array if not isinstance(x, (int, float)) and x is not None]
    print(f"Non-numeric values (first 5, if any): {non_numeric[:5] if non_numeric else 'None'}")

    # Ensure time_index is a DatetimeIndex
    if not isinstance(time_index, pd.DatetimeIndex):
        print("Converting time_index to DatetimeIndex")
        time_index = pd.to_datetime(time_index, errors='coerce')
        if time_index.isna().any():
            print("Error: Some timestamps are invalid")
            return None

    # Create pandas Series
    try:
        inflow_series = pd.Series(inflow_array, index=time_index)
    except Exception as e:
        print(f"Error creating Series: {e}")
        return None

    # Check for NaN in inflow_series
    nan_series_count = inflow_series.isna().sum()
    # Handle NaN values (if any)
    if nan_series_count > 0:
        print("Handling NaN values with linear interpolation")
        inflow_series = inflow_series.interpolate(method='linear')
        inflow_series = inflow_series.fillna(0)  # Fill remaining NaN (e.g., at start/end)
        print(f"Number of NaN values after handling: {inflow_series.isna().sum()}")

    # Verify 30-year average
    mean_inflow = inflow_series.mean()
    print(f"30-year average inflow: {mean_inflow:.4f} (should be ~1)")

    # Calculate percentage deviation from the 30-year average (1)
    percentage_deviation = ((inflow_series - 1) / 1) * 100  # in %
    # Group deviations by year for box plot
    # Create a DataFrame with percentage deviations and year as a column
    deviation_df = pd.DataFrame({
        'deviation': percentage_deviation,
        'year': inflow_series.index.year
    })

    # Prepare data for box plot: group deviations by year
    boxplot_data = [deviation_df[deviation_df['year'] == year]['deviation'].values
                    for year in range(date_start['year'], date_end['year'] + 1)]

    # Plot setup
    import matplotlib
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.serif'] = ['cmr10']  # Computer Modern Roman
    matplotlib.rcParams['axes.formatter.use_mathtext'] = True
    matplotlib.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(15, 6))

    # # Plot yearly inflow
    # ax.plot(time_index, inflow_value, label=f'Inflow ({zone})', color='blue')
    # plt.title(f'Inflow over Time ({zone})')
    # plt.xlabel('Year')
    # plt.ylabel('Normalized Inflow')  # Reflects yearly aggregation
    # plt.legend()
    # plt.tight_layout()
    # plt.grid()
    #
    # # Set x-axis to show every year with 45-degree rotation
    # import matplotlib.dates as mdates
    # ax.xaxis.set_major_locator(mdates.YearLocator(1))  # Every year
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Year format
    # plt.setp(ax.get_xticklabels(), rotation=45)
    # filename = f'InflowProfile_{zone}.pdf'
    # plt.savefig(OUTPUT_PATH_PLOTS / filename)
    # plt.show()
    # Create box plot
    ax.boxplot(boxplot_data, labels=range(date_start['year'], date_end['year'] + 1))
    plt.title(f'Yearly Inflow Deviation from 30-Year Average ({zone})')
    plt.xlabel('Year')
    plt.ylabel('Percentage Deviation (\%)')
    plt.grid(True, axis='y')

    # Rotate x-axis labels by 45 degrees
    plt.setp(ax.get_xticklabels(), rotation=45)

    # Tight layout to prevent label cutoff
    plt.tight_layout()
    plt.show()

    # Calculate and print yearly averages for reference
    yearly_averages = inflow_series.resample('Y').mean()
    yearly_deviations = ((yearly_averages - 1) / 1) * 100
    print("\nYearly Average Inflow and Percentage Deviation from 1:")
    for year, avg, dev in zip(yearly_averages.index.year, yearly_averages, yearly_deviations):
        print(f"{year}: Avg = {avg:.4f}, Deviation = {dev:.2f}%")




def plotInflowValuePlot(data: GridData, zones, OUTPUT_PATH_PLOTS, start, end):
    # Plot setup
    import matplotlib
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.serif'] = ['cmr10']  # Computer Modern Roman
    matplotlib.rcParams['axes.formatter.use_mathtext'] = True
    matplotlib.rcParams['axes.unicode_minus'] = False

    start_date = pd.Timestamp(f"{start['year']}-{start['month']}-{start['day']} {start['hour']}:00",
                              tz='UTC')
    end_date = pd.Timestamp(f"{end['year']}-{end['month']}-{end['day']} {end['hour']}:00", tz='UTC')

    # Create a figure with 5 subplots (one for each zone), stacked vertically
    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(12.405, 17.535), sharex=True)


    # Plot inflow for each zone in its respective subplot
    for i, zone in enumerate(zones):
        # Extract inflow value and time index for the specific zone
        inflow_value = data.profiles[f'inflow_{zone}']  # e.g., 'inflow_NO1'
        time_index = data.profiles['time']

        # Plot in the corresponding subplot
        ax = axes[i]
        ax.plot(time_index, inflow_value, label=f'Inflow ({zone})', color='blue', linewidth=1)
        ax.set_title(f'Inflow over Time ({zone})', fontsize=10)
        ax.set_ylabel('Normalized Inflow', fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True)

        # Set x-axis for the bottom subplot (shared across all subplots)
        ax.set_xlabel('Year', fontsize=10)
        import matplotlib.dates as mdates
        ax.xaxis.set_major_locator(mdates.YearLocator(1))  # Every year
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Year format
        plt.setp(ax.get_xticklabels(), rotation=45)

        # Set x-axis limits to exactly the start and end dates
        ax.set_xlim(start_date, end_date)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the plot (commented out as in your original code)
    filename = 'InflowProfiles_AllZones.pdf'
    plt.savefig(OUTPUT_PATH_PLOTS / filename, dpi=600)
    plt.show()

# === INITIALIZATIONS ===
START = {"year": 1991, "month": 1, "day": 1, "hour": 0}
END = {"year": 2020, "month": 12, "day": 31, "hour": 23}
time_EB = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
zones = ['NO1', 'NO2', 'NO3', 'NO4', 'NO5']
# inflow = plotInflowValueBox(data, zone, OUTPUT_PATH_PLOTS, START, END)
inflow = plotInflowValuePlot(data, zones, OUTPUT_PATH_PLOTS, START, END)


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
END = {"year": 2020, "month": 12, "day": 31, "hour": 23}
time_Shed = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
dfplot = plotEnergyMix(data=data, database=database, areas=['NO', 'SE', 'FI', 'DK'], timeMaxMin=time_Shed, variable="capacity")

# %% === GET ENERGY BALANCE ON NODAL AND ZONAL LEVEL ===

START = {"year": 1991, "month": 1, "day": 1, "hour": 0}
END = {"year": 1991, "month": 12, "day": 31, "hour": 23}
time_EB = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
totalProduction = getProductionForAllNodesFromDBTest(data, database, time_EB)

# %%

energyBalance = {}

for i in range(0, 30):
    year = 1991 + i
    print(year)

    START = {"year": year, "month": 1, "day": 1, "hour": 0}
    END = {"year": year, "month": 12, "day": 31, "hour": 23}
    time_EB = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
    all_nodes = data.node.id

    totalProduction = getProductionForAllNodesFromDBTest(data, database, time_EB)
    flow_data = getFlowDataOnALLBranchesTest(data, database, time_EB)



    #flow_data = getFlowDataOnALLBranches(data, database, time_EB)       # TAR LANG TID
    totalDemand = getDemandForAllNodesFromDB(data, database, time_EB)
    #totalProduction = getProductionForAllNodesFromDB(data, database, time_EB)   # TAR LANG TID
    totalLoadShedding = database.getResultLoadheddingSum(timeMaxMin=time_EB)

    # Calculate energy balance at node and zone levels
    node_energyBalance = getEnergyBalanceNodeLevel(all_nodes, totalDemand, totalProduction, totalLoadShedding, flow_data, OUTPUT_PATH, VERSION, START)
    zone_energyBalance = getEnergyBalanceZoneLevel(all_nodes, totalDemand, totalProduction, totalLoadShedding, flow_data, OUTPUT_PATH, VERSION, START)
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
country = "FI"  # Country code

n_ideal_years = 30
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



# %% === ZONAL PRICE MAP ===
# TODO: legg til mulighet for å ha øre/kwh
zones = ['NO1', 'NO2', 'NO3', 'NO4', 'NO5', 'SE1', 'SE2', 'SE3', 'SE4',
         'DK1', 'DK2', 'FI', 'DE', 'GB', 'NL', 'LT', 'PL', 'EE']
year_range = list(range(SIM_YEAR_START, SIM_YEAR_END + 1))
price_matrix, log = createZonePriceMatrix(data, database, zones, year_range, TIMEZONE, SIM_YEAR_START, SIM_YEAR_END)

# Plot Zonal Price Matrix
plotZonePriceMatrix(price_matrix, save_fig=True, OUTPUT_PATH_PLOTS=OUTPUT_PATH_PLOTS, start=START, end=END, version=VERSION)



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
END = {"year": 1993, "month": 12, "day": 31, "hour": 23}

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
END = {"year": 1993, "month": 12, "day": 31, "hour": 23}

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


# %%

# === INITIALIZATIONS ===
START = {"year": 1993, "month": 1, "day": 1, "hour": 0}
END = {"year": 1993, "month": 12, "day": 31, "hour": 23}
area = None
zone = 'NO2'

# Juster area for å se på sonene, og zone for å se på nodene i sonen
# === COMPUTE TIMERANGE AND PLOT FLOW ===
time_Prod = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
correct_date_start_Prod = DATE_START + pd.Timedelta(hours=time_Prod[0])
if area is not None:
    zones_in_area_prod = getProductionZonesInArea(data, database, area, time_Prod, OUTPUT_PATH, correct_date_start_Prod, week=True)
    energyBalanceZones = zones_in_area_prod.sum(axis=0)

if zone is not None:
    nodes_in_zone_prod = getProductionNodesInZone(data, database, zone, time_Prod, OUTPUT_PATH, correct_date_start_Prod, week=True)
    energyBalanceNodes = nodes_in_zone_prod.sum(axis=0)



#%% Excel ###
"""
Production, consumption, and price data for specific nodes within a given time period.

Main Features:
- Handles time using Python's built-in datetime objects.
- Retrieves simulated production, consumption, and price data from a given SQL file for selected nodes within a specified timeframe.
- Organizes data and exports it to an Excel file for further analysis.
"""

# === INITIALIZATIONS ===
START = {"year": 1991, "month": 4, "day": 14, "hour": 0}
END = {"year": 1991, "month": 5, "day": 31, "hour": 23}
Nodes = ["DK1_2", "DK1_3", "FI_10", "FI_12", "SE3_3", "SE3_6"]
SELECTED_BRANCHES  = [['SE3_1','FI_10'], ['SE3_3', 'FI_10']]
# ======================================================================================================================

start_hour, end_hour = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
production_per_node, gen_idx, gen_type = GetProductionAtSpecificNodes(Nodes, data, database, start_hour, end_hour)


consumption_per_node = GetConsumptionAtSpecificNodes(Nodes, data, database, start_hour, end_hour)
nodal_prices_per_node = GetPriceAtSpecificNodes(Nodes, data, database, start_hour, end_hour)
reservoir_filling_per_node, storage_cap = GetReservoirFillingAtSpecificNodes(Nodes, data, database, start_hour, end_hour)
flow_data = getFlowDataOnBranches(data, database, [start_hour, end_hour], SELECTED_BRANCHES)
excel_filename = ExportToExcel(Nodes, production_per_node, consumption_per_node, nodal_prices_per_node, reservoir_filling_per_node, storage_cap, flow_data, START, END, SCENARIO, VERSION, OUTPUT_PATH)




# %% === PDF HANDLING ===

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib

# Enable LaTeX rendering for Computer Modern font
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['cmr10']  # Computer Modern Roman
matplotlib.rcParams['axes.formatter.use_mathtext'] = True  # Fix cmr10 warning
matplotlib.rcParams['axes.unicode_minus'] = False  # Fix minus sign rendering

# Placeholder: Define zones (excluding 'DE') based on previous node-to-zone mapping
node_names = [node for node in all_nodes]
node_to_zone = {}
for node in node_names:
    if '_' in node:
        zone = node.split('_')[0]
    else:
        zone = node
    node_to_zone[node] = zone
all_zones = sorted([zone for zone in set(node_to_zone.values())])

# Placeholder: Generate sample data for each zone (replace with actual data)
time_period = pd.date_range(start="2025-01-01", end="2025-01-02", freq="h")  # Use 'h' for hourly
n_timesteps = len(time_period)

# Sample data structure
zone_data = {}
for zone in all_zones:
    zone_data[zone] = {
        'consumption': pd.DataFrame({
            'time': time_period,
            'demand': np.random.uniform(100, 1000, n_timesteps)  # MW
        }),
        'generation': pd.DataFrame({
            'time': time_period,
            'wind': np.random.uniform(0, 500, n_timesteps),      # MW
            'hydro': np.random.uniform(0, 600, n_timesteps),     # MW
            'thermal': np.random.uniform(0, 400, n_timesteps)    # MW
        }),
        'prices': pd.DataFrame({
            'time': time_period,
            'price': np.random.uniform(20, 100, n_timesteps)     # €/MWh
        }),
        'storage': pd.DataFrame({
            'time': time_period,
            'level': np.random.uniform(0, 1000, n_timesteps)     # MWh
        }) if np.random.rand() > 0.3 else None  # Simulate some zones lacking storage
    }

# PDF settings
pdf_filename = "zone_energy_report.pdf"
page_width, page_height = 8.27, 11.69  # A4 in inches (595 x 842 points at 72 DPI)
margin = 0.25  # Reduced margins (in inches)
plot_width = page_width - 2 * margin  # ~7.77 inches
plot_height = 3.0  # Plot height (in inches, ~216 points)
spacing = 0.2  # Spacing between plots (in inches)
max_page_height = page_height - 2 * margin  # Usable page height (~11.19 inches)

# Initialize PDF
with PdfPages(pdf_filename) as pdf:
    for zone in all_zones:
        current_height = 0  # Track total height used on current page
        figs = []  # Store figures for this zone

        # 1. Consumption Plot
        fig, ax = plt.subplots(figsize=(plot_width, plot_height))
        data = zone_data[zone]['consumption']
        ax.plot(data['time'], data['demand'], color='red', label='Demand')
        ax.set_title(f"Consumption in {zone}", fontsize=12)
        ax.set_xlabel("Time", fontsize=10)
        ax.set_ylabel("Demand (MW)", fontsize=10)
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45, fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        figs.append(fig)

        # 2. Generation by Type Plot
        fig, ax = plt.subplots(figsize=(plot_width, plot_height))
        gen_data = zone_data[zone]['generation']
        for column in gen_data.columns[1:]:  # Skip 'time'
            ax.plot(gen_data['time'], gen_data[column], label=column.capitalize())
        ax.set_title(f"Generation by Type in {zone}", fontsize=12)
        ax.set_xlabel("Time", fontsize=10)
        ax.set_ylabel("Generation (MW)", fontsize=10)
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45, fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        figs.append(fig)

        # 3. Prices Plot
        fig, ax = plt.subplots(figsize=(plot_width, plot_height))
        price_data = zone_data[zone]['prices']
        ax.plot(price_data['time'], price_data['price'], color='blue', label='Price')
        ax.set_title(f"Prices in {zone}", fontsize=12)
        ax.set_xlabel("Time", fontsize=10)
        ax.set_ylabel("Price (€/MWh)", fontsize=10)
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45, fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        figs.append(fig)

        # 4. Storage Filling Plot (if available)
        if zone_data[zone]['storage'] is not None:
            fig, ax = plt.subplots(figsize=(plot_width, plot_height))
            storage_data = zone_data[zone]['storage']
            ax.plot(storage_data['time'], storage_data['level'], color='purple', label='Storage Level')
            ax.set_title(f"Storage Filling in {zone}", fontsize=12)
            ax.set_xlabel("Time", fontsize=10)
            ax.set_ylabel("Storage (MWh)", fontsize=10)
            ax.legend()
            ax.grid(True)
            plt.xticks(rotation=45, fontsize=8)
            plt.yticks(fontsize=8)
            plt.tight_layout()
            figs.append(fig)

        # Save figures to PDF, grouping by zone
        for i, fig in enumerate(figs):
            # Calculate height needed for this plot (plot_height + spacing, except for last plot)
            plot_total_height = plot_height + (spacing if i < len(figs) - 1 else 0)

            # Check if plot fits on current page
            if current_height + plot_total_height > max_page_height and current_height > 0:
                pdf.savefig(bbox_inches='tight')  # Save current page
                current_height = 0  # Reset for new page

            pdf.savefig(fig, bbox_inches='tight')  # Save figure
            current_height += plot_total_height  # Update height used
            plt.close(fig)  # Close figure to free memory

        # Reset for next zone (no extra pdf.savefig needed)
        current_height = 0

print(f"PDF generated: {pdf_filename}")