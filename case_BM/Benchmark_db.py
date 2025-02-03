from functions.global_functions import *  # Functions like 'read_grid_data', 'solve_lp' m.m.
from functions.database_functions import  * # Functions like 'getSystemCostFromDB' m.m.


YEAR_SCENARIO = 2025
YEAR_START = 1991
YEAR_END = 2020
case = 'BM'
version = 'v3'

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

x = 1

# Example usage:
timeMaxMin_YEAR = get_time_steps_for_period(2000, 2000)
print(f"Time steps for 2000: {timeMaxMin_YEAR}")
list_of_years = get_time_steps_for_years(selected_years=[1993, 2001, 2009, 2018])
print(f"Time steps for years: {list_of_years}")

# %%

time_SC = get_time_steps_for_period(2000, 2000) # eller time_max_min
print(f"System cost {sum(getSystemCostFromDB(data=data, db=database, timeMaxMin=time_SC).values()):.2f} EUR, or {sum(getSystemCostFromDB(data=data, db=database, timeMaxMin=time_SC).values())/1e9:.2f} Billion EUR")

time_MP = get_time_steps_for_period(2000, 2000) # eller time_max_min
print(f"Mean area price {sum(getAreaPricesAverageFromDB(data=data, db=database, areas=None, timeMaxMin=time_MP).values()) / len(getAreaPricesAverageFromDB(data=data, db=database, areas=None, timeMaxMin=time_MP)):.2f} EUR/MWh")



# %% Map prices and branch utilization
output_path = OUTPUT_PATH / f'prices_and_branch_utilization_map_{version}.html'

time_map = get_time_steps_for_period(2000, 2000)
create_price_and_utilization_map_FromDB(data, database, time_max_min=time_map, output_path=output_path)


# %%

storfilling = pd.DataFrame()
areas = ["NO"]          # When plotting multiple years in one year, recommend to only use one area
relative=True           # Relative storage filling, True gives percentage
interval=1              # Month interval for x-axis if plot_by_year is False
plot_by_year = True    # True: Split plot by year, or False: Plot all years in one plot
duration_curve = False  # True: Plot duration curve, or False: Plot storage filling over time
save_plot_SF = False    # True: Save plot as pdf

time_SF = get_time_steps_for_period(2000, 2001) # eller time_max_min

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






# %% Plot nodal prices Norway
zone = 'NO1'

plot_all_nodes = False                   # Plot nodal prices for all nodes in the zone(True) or average(False)
interval = 12                           # Month interval for x-axis if plot_by_year_nodal is False
save_plot_nodal = True                 # Save plot as pdf
plot_by_year_nodal = False              # Plot all years in one plot(False) or split plot by year(True)
duration_curve_nodal = False            # Plot duration curve(False) or nodal prices over year(True)

time_NP = get_time_steps_for_period(2000, 2001)

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


# %% Hydro production, reservoir filling, inflow
area_OP = 'NO'
genType = 'hydro'
save_plot_HRI = False
interval = 12
box_in_frame = True
plot_full_timeline = True       # Plot full timeline or by year
relative_storage = True         # Relative storage filling, True gives percentage

time_HRI = get_time_steps_for_period(2000, 2001)

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



# %% Plot nodal prices, demand and hydro production

area_OP = 'NO'
title = "Avg. Area Price, Demand and Hydro Production in NO"
plot_full_timeline = True
save_fig_PDP = False
interval = 1
box_in_frame = True
resample = True

time_PLP = get_time_steps_for_period(2000, 2001)
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




# %% Load, generation by type in AREA
area_OP = 'NO'
title_gen = f'Production, Consumption and Price in {area_OP}'
interval = 12
figsize = (10, 6)

time_LGT = get_time_steps_for_period(2000, 2003)
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


