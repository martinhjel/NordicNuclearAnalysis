from functions.global_functions import *  # Functions like 'read_grid_data', 'solve_lp' m.m.
from functions.database_functions import  * # Functions like 'getSystemCostFromDB' m.m.


YEAR_SCENARIO = 2025
YEAR_START = 1991
YEAR_END = 2020
case = 'BM'
version = '52_v17'

# DATE_START = f"{YEAR_START}-01-01"
DATE_START = pd.Timestamp(f'{YEAR_START}-01-01 00:00:00', tz='UTC')
# DATE_END = f"{YEAR_END}-12-31"
DATE_END = pd.Timestamp(f'{YEAR_END}-12-31 23:00:00', tz='UTC')
loss_method = 0
new_scenario = False
save_scenario = False


# Get the base directory
try:
    # For scripts
    BASE_DIR = pathlib.Path(__file__).parent
except NameError:
    # For notebooks or interactive shells
    BASE_DIR = pathlib.Path().cwd()
    BASE_DIR = BASE_DIR / f'case_{case}'


SQL_FILE = BASE_DIR / f"powergama_{case}_{version}.sqlite"

# File paths
DATA_PATH = BASE_DIR / 'data'
OUTPUT_PATH = BASE_DIR / 'results'
OUTPUT_PATH_PLOTS = BASE_DIR / 'results' / 'plots'


data, time_max_min = setup_grid(YEAR_SCENARIO, version, DATE_START, DATE_END, DATA_PATH, new_scenario, save_scenario)
database = Database(SQL_FILE)


# # Example usage:
# timeMaxMin_YEAR = get_time_steps_for_period(2000, 2000)
# print(f"Time steps for 2000: {timeMaxMin_YEAR}")
# list_of_years = get_time_steps_for_years(selected_years=[1993, 2001, 2009, 2018])
# print(f"Time steps for years: {list_of_years}")



# %%
# time_max_min #
def calcSystemCostAndMeanPriceFromDB(data: GridData, database: Database, time_max_min):
    time_SC = time_max_min #get_time_steps_for_period(2000, 2000) # eller time_max_min
    print(f"System cost {sum(getSystemCostFromDB(data=data, db=database, timeMaxMin=time_SC).values()):.2f} EUR, or {sum(getSystemCostFromDB(data=data, db=database, timeMaxMin=time_SC).values())/1e9:.2f} Billion EUR")

    time_MP = time_max_min #get_time_steps_for_period(2000, 2000) # eller time_max_min
    print(f"Mean area price {sum(getAreaPricesAverageFromDB(data=data, db=database, areas=None, timeMaxMin=time_MP).values()) / len(getAreaPricesAverageFromDB(data=data, db=database, areas=None, timeMaxMin=time_MP)):.2f} EUR/MWh")


# calcSystemCostAndMeanPriceFromDB(data, database, time_max_min)



# %% Map prices and branch utilization
def plot_Map(data: GridData, database: Database, time_max_min, OUTPUT_PATH, version):
    output_path = OUTPUT_PATH / f'prices_and_branch_utilization_map_{version}.html'

    time_map = time_max_min #get_time_steps_for_period(2000, 2000)
    create_price_and_utilization_map_FromDB(data, database, time_max_min=time_map, output_path=output_path)

# plot_Map(data, database, time_max_min, OUTPUT_PATH, version)

# %% Storage Filling
# OBS OBS, sjekk timeMaxMin, hva den er, husk timeMaxMin henter ut data fra timeseries_profile

def calcPlot_SF_Areas_FromDB(data: GridData, database: Database, time_max_min, OUTPUT_PATH_PLOTS, DATE_START):
    storfilling = pd.DataFrame()
    areas = ["NO"]          # When plotting multiple years in one year, recommend to only use one area
    relative=True           # Relative storage filling, True gives percentage
    interval=1              # Month interval for x-axis if plot_by_year is False
    plot_by_year = True    # True: Split plot by year, or False: Plot all years in one plot
    duration_curve = False  # True: Plot duration curve, or False: Plot storage filling over time
    save_plot_SF = False    # True: Save plot as pdf

    time_SF = get_time_steps_for_period(2015, 2015) # eller time_max_min

    for area in areas:
        storfilling[area] = getStorageFillingInAreasFromDB(data=data,
                                                           db=database,
                                                           areas=[area],
                                                           generator_type="hydro",
                                                           relative_storage=relative,
                                                           timeMaxMin=time_SF)
        if relative:
            storfilling[area] = storfilling[area] * 100

    # Compute the correct DATE_START for this year
    correct_date_start_SF = DATE_START + pd.Timedelta(hours=time_SF[0])
    correct_date_end_SF = DATE_START + pd.Timedelta(hours=time_SF[-1])

    storfilling.index = pd.date_range(correct_date_start_SF, periods=time_SF[-1] - time_SF[0], freq='h')
    storfilling['year'] = storfilling.index.year    # Add year column to DataFrame
    title_storage_filling = f'Reservoir Filling in {areas} for period {correct_date_start_SF.year}-{correct_date_end_SF.year}'
    plot_storage_filling_area(storfilling=storfilling,
                              DATE_START=correct_date_start_SF,
                              DATE_END=correct_date_end_SF,
                              areas=areas,
                              interval=interval,
                              title=title_storage_filling,
                              OUTPUT_PATH_PLOTS=OUTPUT_PATH_PLOTS,
                              relative=relative,
                              plot_by_year=plot_by_year,
                              save_plot=save_plot_SF,
                              duration_curve=duration_curve,
                              tex_font=False)


calcPlot_SF_Areas_FromDB(data, database, time_max_min, OUTPUT_PATH_PLOTS, DATE_START)



def calcPlot_SF_Zones_FromDB(data: GridData, database: Database, time_max_min, OUTPUT_PATH_PLOTS, DATE_START):
    storfilling = pd.DataFrame()
    zones = ['NO1', 'NO2', 'NO3', 'NO4', 'NO5']   # When plotting multiple years in one year, recommend to only use one area
    relative=True           # Relative storage filling, True gives percentage
    interval=1              # Month interval for x-axis if plot_by_year is False
    plot_by_year = False    # True: Split plot by year, or False: Plot all years in one plot
    duration_curve = False  # True: Plot duration curve, or False: Plot storage filling over time
    save_plot_SF = False    # True: Save plot as pdf

    time_SF = get_time_steps_for_period(2015, 2015) # eller time_max_min

    for zone in zones:
        storfilling[zone] = getStorageFillingInZonesFromDB(data=data,
                                                           db=database,
                                                           zones=[zone],
                                                           generator_type="hydro",
                                                           relative_storage=relative,
                                                           timeMaxMin=time_SF)
        if relative:
            storfilling[zone] = storfilling[zone] * 100

    # Compute the correct DATE_START for this year
    correct_date_start_SF = DATE_START + pd.Timedelta(hours=time_SF[0])
    correct_date_end_SF = DATE_START + pd.Timedelta(hours=time_SF[-1])

    storfilling.index = pd.date_range(correct_date_start_SF, periods=time_SF[-1] - time_SF[0], freq='h')
    storfilling['year'] = storfilling.index.year    # Add year column to DataFrame
    title_storage_filling = f'Reservoir Filling in {zones} for period {correct_date_start_SF.year}-{correct_date_end_SF.year}'
    plot_storage_filling_area(storfilling=storfilling,
                              DATE_START=correct_date_start_SF,
                              DATE_END=correct_date_end_SF,
                              areas=zones,
                              interval=interval,
                              title=title_storage_filling,
                              OUTPUT_PATH_PLOTS=OUTPUT_PATH_PLOTS,
                              relative=relative,
                              plot_by_year=plot_by_year,
                              save_plot=save_plot_SF,
                              duration_curve=duration_curve,
                              tex_font=False)

 calcPlot_SF_Zones_FromDB(data, database, time_max_min, OUTPUT_PATH_PLOTS, DATE_START)



# %% Plot nodal prices Norway
def calcPlot_NP_FromDB(data: GridData, database: Database, time_max_min, OUTPUT_PATH_PLOTS, DATE_START):
    zone = 'NO2'

    plot_all_nodes = False                   # Plot nodal prices for all nodes in the zone(True) or average(False)
    interval = 1                           # Month interval for x-axis if plot_by_year_nodal is False
    save_plot_nodal = True                 # Save plot as pdf
    plot_by_year_nodal = False              # Plot all years in one plot(False) or split plot by year(True)
    duration_curve_nodal = False            # Plot duration curve(False) or nodal prices over year(True)

    time_NP = time_max_min# get_time_steps_for_period(2000, 2001)

    nodes_in_zone = data.node[data.node['zone'] == zone].index.tolist() # Get all nodes in the zone
    # Get nodal prices for all nodes in the zone in one step node_prices
    node_prices = pd.DataFrame({node: getNodalPricesFromDB(database, node, time_NP) for node in nodes_in_zone})#  * EUR_MWH_TO_ORE_KWH

    correct_date_start_NP = DATE_START + pd.Timedelta(hours=time_NP[0])
    correct_date_end_NP = DATE_START + pd.Timedelta(hours=time_NP[-1])
    node_prices.index = pd.date_range(correct_date_start_NP, periods=time_NP[-1] - time_NP[0], freq='h')
    title_nodal = f"Avg. Prices in {zone} for period {correct_date_start_NP.year}-{correct_date_end_NP.year}"
    # title_nodal = f"Nodal Prices in {zone} for period {YEAR_START}-{YEAR_END}"
    plot_nodal_prices_FromDB(data=data,
                             node_prices=node_prices,
                             nodes_in_zone=nodes_in_zone,
                             zone=zone,
                             DATE_START=correct_date_start_NP,
                             DATE_END=correct_date_end_NP,
                             interval=interval,
                             TITLE=title_nodal,
                             plot_all_nodes=plot_all_nodes,
                             save_plot_nodal=save_plot_nodal,
                             OUTPUT_PATH_PLOTS=OUTPUT_PATH_PLOTS,
                             plot_by_year_nodal=plot_by_year_nodal,
                             duration_curve_nodal=duration_curve_nodal,
                             tex_font=False)

# calcPlot_NP_FromDB(data, database, time_max_min, OUTPUT_PATH_PLOTS, DATE_START)


# %% Hydro production, reservoir filling, inflow
def calcPlot_HRI_FromDB(data: GridData, database: Database, time_max_min, OUTPUT_PATH_PLOTS, DATE_START):
    area_OP = 'NO'
    genType = 'hydro'
    save_plot_HRI = False
    interval = 1
    box_in_frame = True
    plot_full_timeline = True       # Plot full timeline or by year
    relative_storage = True         # Relative storage filling, True gives percentage

    time_HRI = time_max_min# get_time_steps_for_period(2000, 2001)

    correct_date_start_HRI = DATE_START + pd.Timedelta(hours=time_HRI[0])
    correct_date_end_HRI = DATE_START + pd.Timedelta(hours=time_HRI[-1])

    df_resampled = calculate_Hydro_Res_Inflow_FromDB(data,
                                                     database,
                                                     correct_date_start_HRI,
                                                     area_OP,
                                                     genType,
                                                     time_HRI,
                                                     relative_storage,
                                                     include_pump=False)
    df_resampled['year'] = df_resampled.index.year
    title_HRI = 'Hydro Production, Reservoir Filling and Inflow'


    plot_hydro_prod_res_inflow(df=df_resampled,
                               DATE_START=correct_date_start_HRI,
                               DATE_END=correct_date_end_HRI,
                               interval=interval,
                               TITLE=title_HRI,
                               OUTPUT_PATH_PLOTS=OUTPUT_PATH_PLOTS,
                               save_plot=save_plot_HRI,
                               box_in_frame=box_in_frame,
                               plot_full_timeline=plot_full_timeline,
                               tex_font=False)

# calcPlot_HRI_FromDB(data, database, time_max_min, OUTPUT_PATH_PLOTS, DATE_START)


# %% Plot nodal prices, demand and hydro production

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

# calcPlot_PLP_FromDB(data, database, time_max_min, OUTPUT_PATH_PLOTS, DATE_START)


# %% Load, generation by type in AREA
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

# area='FI'
# df_gen_re, df_prices, tot_prod = calcPlot_LG_FromDB(data, database, time_max_min, OUTPUT_PATH_PLOTS, DATE_START, area_OP=area)
# print(f"Total production in {area}: {tot_prod:.2f} MWh")
#
#
# # %%
# df_gen_re['Total Generation'] = df_gen_re.drop(columns=['Load']).sum(axis=1)
#
# df_gen_re[['Total Generation', 'Load']].plot(figsize=(12, 6), linewidth=2)
# plt.title("Total Generation vs. Load Over Time")
# plt.ylabel("Energy (MWh)")
# plt.xlabel("Date")
# plt.legend(loc='upper left', bbox_to_anchor=(1,1))
# plt.grid(True)
# plt.show()
#
#
# # %%
#
#
# df_gen_re.drop(columns=['Load', 'Total Generation'], errors='ignore').sum().plot(kind='bar',
#                                                                                  figsize=(10, 6),
#                                                                                  color='skyblue',
#                                                                                  edgecolor='black')
# plt.title("Total Energy Generation by Type")
# plt.ylabel("Total Energy (MWh)")
# plt.xlabel("Energy Source")
# plt.xticks(rotation=45)
# plt.grid(axis='y')
# plt.show()


# %%


def getProductionNodeAndZones(data: GridData, db: Database, area, zone, time_max_min, DATE_START):

    zones_in_area_prod = get_production_by_type_FromDB_ZoneLevel(data, db, area=area, time_max_min=time_max_min, DATE_START=DATE_START)
    zones_in_area_prod.to_csv(f'production_zone_level_{area}_{DATE_START}.csv')

    nodes_in_zone_prod = get_production_by_type_FromDB_NodesInZone(data, db, zone=zone, time_max_min=time_max_min, DATE_START=DATE_START)
    nodes_in_zone_prod.to_csv(f'production_nodes_in_zone_{zone}_{DATE_START}.csv')

    return zones_in_area_prod, nodes_in_zone_prod

# Juster area for å se på sonene, og zone for å se på nodene i sonen
zones_in_area_prod, nodes_in_zone_prod = getProductionNodeAndZones(data, database, area='NO', zone='NO2', time_max_min=time_max_min, DATE_START=DATE_START)


# %% Total production and load
# print(f"Total production: {total_production/1e6:.2f} TWh")
# load_demand = data.getDemandPerArea(area='NO')
# print(f"Total load: {sum(load_demand['sum'])/1e6:.2f} TWh")

spillage = ['wind_on', 'wind_off', 'solar', 'hydro', 'ror', 'nuclear']
energy_balance = getEnergyBalanceInAreaFromDB(data, db=database, area='NO', spillageGen=spillage, timeMaxMin=time_max_min, start_date=DATE_START)

# print(f"Energy balance: {energy_balance:.2f} TWh")