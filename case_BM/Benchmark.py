# Imports
from powergama.database import Database  # Import Database-Class specifically
from functions.global_functions import *
from scripts.case_doc import *


# Define global variables
YEAR_SCENARIO = 2025
case = 'BM'
version = '52_v21'
YEAR_START = 1991
YEAR_END = 2020

# SQL_FILE = "powergama_2025_30y_v1.sqlite"
# DATE_START = f"{YEAR_START}-01-01"
DATE_START = pd.Timestamp(f'{YEAR_START}-01-01 00:00:00', tz='UTC')

# DATE_END = f"{YEAR_END}-01-02"
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


# week_start = np.random.permutation([16, 20, 24, 28, 32, 36])  # Middle of Apr, May, Jun, Jul, Aug, Sep
# Randomised week sequence, allocated to NNP in indexed order,
# meaning first index in week_start is allocated to first indexed NNP.
# This means finland gets first go.
# week_start = [16, 36, 20, 24, 32, 28]
# week Maintenance Start Order
week_MSO = {'FI_10':16,
            'FI_12':36,
            'SE3_3':20,
            'SE3_6':24,
            'GB':32,
            'NL':28
            }

# %%
# Configure grid and run simulation
create_case_doc('BM') # Create case documentation
data, time_max_min = setup_grid(YEAR_SCENARIO, version, DATE_START, DATE_END, DATA_PATH, new_scenario, save_scenario)
res = solve_lp(data, SQL_FILE, loss_method, replace=True, nuclear_availability=0.7, week_MSO=week_MSO)

res.getEnergyBalanceInArea(area='NO', spillageGen='wind_on')

# %% Print results
print(f"System cost {sum(res.getSystemCost().values()):.2f} EUR, or {sum(res.getSystemCost().values())/1e9:.2f} Billion EUR")
print(f"Mean area price {sum(res.getAreaPricesAverage().values()) / len(res.getAreaPricesAverage()):.2f} EUR/MWh")
# %%

# Database instance
database = Database(SQL_FILE)
grid_data_path = DATA_PATH / 'system'
# %%

output_path = OUTPUT_PATH / f'prices_and_branch_utilization_map_{version}.html'
create_price_and_utilization_map(data, res, time_max_min=time_max_min, output_path=output_path)


# %% Plot import/ export load flow with respect to time for all cross-border interconnections

# time_max_min = [0, 24]
by_year = False
duration_curve = True
duration_relative = True   # Hours(False) or Percentage(True)
save_fig_flow = False
interval_flow = 12           # Velger antall måneder mellom hver x-akse tick
plot_imp_exp_cross_border_Flow_NEW(db=database,
                                   DATE_START=DATE_START,
                                   time_max_min=time_max_min,
                                   grid_data_path=grid_data_path,
                                   OUTPUT_PATH_PLOTS=OUTPUT_PATH_PLOTS,
                                   by_year=by_year,
                                   duration_curve=duration_curve,
                                   duration_relative=duration_relative,
                                   save_fig=save_fig_flow,
                                   interval=interval_flow,
                                   check=False,
                                   tex_font=False)


# %% Plot storage filling levels
storfilling = pd.DataFrame()
areas = ["NO"]          # When plotting multiple years in one year, recommend to only use one area
relative=True           # Relative storage filling, True gives percentage
interval=12              # Month interval for x-axis if plot_by_year is False
plot_by_year = True    # True: Split plot by year, or False: Plot all years in one plot
duration_curve = False  # True: Plot duration curve, or False: Plot storage filling over time
save_plot_SF = False    # True: Save plot as pdf

for area in areas:
    storfilling[area] = res.getStorageFillingInAreas(areas=[area],
                                                     generator_type="hydro",
                                                     relative_storage=relative)
    if relative:
        storfilling[area] = storfilling[area] * 100

storfilling.index = pd.date_range(DATE_START, periods=time_max_min[-1], freq='h')
storfilling['year'] = storfilling.index.year    # Add year column to DataFrame
title_storage_filling = f'Reservoir Filling in {areas} for period {DATE_START.year}-{DATE_END.year}'
plot_storage_filling_area(storfilling=storfilling,
                          DATE_START=DATE_START,
                          DATE_END=DATE_END,
                          areas=areas,
                          interval=interval,
                          title=title_storage_filling,
                          OUTPUT_PATH_PLOTS=OUTPUT_PATH_PLOTS,
                          relative=relative,
                          plot_by_year=plot_by_year,
                          save_plot=save_plot_SF,
                          duration_curve=duration_curve,
                          tex_font=False)

tex_font = False
save_plot = True

fig, ax = plt.subplots(figsize=(10, 6))
# Plot logic
if not plot_by_year:
    for area in areas:
        if area in storfilling.columns:
            ax.plot(storfilling.index, storfilling[area], label=f"{area}")
        else:
            raise ValueError(f"{area} not found in storfilling DataFrame columns")

    # Configure axes for date-based plotting
    configure_axes(ax, relative, x_label='Date')
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    ax.set_xlim(pd.to_datetime(DATE_START), pd.to_datetime(DATE_END))

    # Add legend and title
    lines, labels = ax.get_legend_handles_labels()
    ax.legend(lines, labels, loc='upper left')


else:
    for year in storfilling['year'].unique():
        for area in areas:
            if area in storfilling.columns:
                group = storfilling[storfilling['year'] == year]
                # ax.plot(group.index.dayofyear, group[area], label=f"{year}")
                if duration_curve:
                    # Duration curve: sort values in descending order
                    sorted_values = group[area].sort_values(ascending=False).reset_index(drop=True)
                    ax.plot(sorted_values, label=f"{year} (Duration Curve)")
                else:
                    # Standard plot with day of year
                    ax.plot(group.index.dayofyear, group[area], label=f"{year}")
            else:
                raise ValueError(f"{area} not found in storfilling DataFrame columns")


    # Configure axes for yearly or duration curve plotting
    configure_axes(ax, relative, x_label='Date' if not duration_curve else 'Hours')
    if not duration_curve:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        ax.set_xlim(0, 364)

    if relative:
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
        ax.set_ylim(0, 100)

    # Add legend and title
    lines, labels = ax.get_legend_handles_labels()
    print("Lines:", lines)
    print("Labels:", labels)

    # Place legend below the plot
    ax.legend(lines, labels,
               loc='upper center', bbox_to_anchor=(0.5, -0.15),
               ncol=8, frameon=False)  # Adjust ncol as needed to fit all items
    # plt.subplots_adjust(bottom=0.2)  # Make space for the legend below the plot



plt.title(title_storage_filling)
plt.grid(True)
plt.tight_layout()
plt.subplots_adjust(bottom=0.3)  # Reserve space for legend
if tex_font:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"]})

# Show and/or save the plot
if save_plot:
    plt.savefig(OUTPUT_PATH_PLOTS / 'storage_level.pdf')
plt.show()




# %% Plot nodal prices Norway
zone = 'NO1'

plot_all_nodes = False                   # Plot nodal prices for all nodes in the zone(True) or average(False)
interval = 12                           # Month interval for x-axis if plot_by_year_nodal is False
save_plot_nodal = True                 # Save plot as pdf
plot_by_year_nodal = False              # Plot all years in one plot(False) or split plot by year(True)
duration_curve_nodal = False            # Plot duration curve(False) or nodal prices over year(True)

nodes_in_zone = res.grid.node[res.grid.node['zone'] == zone].index.tolist() # Get all nodes in the zone
# Get nodal prices for all nodes in the zone in one step node_prices
node_prices = pd.DataFrame({node: res.getNodalPrices(node=node) for node in nodes_in_zone})#  * EUR_MWH_TO_ORE_KWH
node_prices.index = pd.date_range(DATE_START, periods=time_max_min[-1], freq='h')
title_nodal = f"Avg. Prices in {zone} for period {YEAR_START}-{YEAR_END}"
# title_nodal = f"Nodal Prices in {zone} for period {YEAR_START}-{YEAR_END}"
plot_nodal_prices(res=res,
                  node_prices=node_prices,
                  nodes_in_zone=nodes_in_zone,
                  zone=zone,
                  DATE_START=DATE_START,
                  DATE_END=DATE_END,
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
save_plot_HRI = True
interval = 12
box_in_frame = True
plot_full_timeline = True       # Plot full timeline or by year
df_resampled = calculate_Hydro_Res_Inflow(res,
                                          data,
                                          DATE_START,
                                          area_OP,
                                          genType,
                                          time_max_min,
                                          include_pump=False)
df_resampled['year'] = df_resampled.index.year
title_HRI = 'Hydro Production, Reservoir Filling and Inflow'
plot_hydro_prod_res_inflow(df=df_resampled,
                           DATE_START=DATE_START,
                           DATE_END=DATE_END,
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
df_plp, df_plp_resampled = calc_PLP(res, area_OP, DATE_START, time_max_min)
plot_hydro_prod_demand_price(df_plp=df_plp,
                             df_plp_resampled=df_plp_resampled,
                             resample=resample,
                             DATE_START=DATE_START,
                             DATE_END=DATE_END,
                             interval=interval,
                             TITLE=title,
                             save_fig=save_fig_PDP,
                             plot_full_timeline=plot_full_timeline,
                             box_in_frame=box_in_frame,
                             OUTPUT_PATH_PLOTS=OUTPUT_PATH_PLOTS,
                             tex_font=False)


# %% Loadshedding at all nodes for a given year
year = 0 # 0-indexed from start year
start_ls = 8759*(year)
end_ls = 8759*(year+1)
load_shedding = pd.Series(res.db.getResultLoadheddingSum(timeMaxMin=[start_ls, end_ls]))
check_load_shedding(load_shedding, tex_font=False)

# %% Load, generation by type in AREA
area_OP = 'NO'
title_gen = f'Production, Consumption and Price in {area_OP}'
interval = 12
figsize = (10, 6)
df_gen_resampled, df_prices_resampled, total_production = get_production_by_type(res, area_OP, time_max_min, DATE_START)
plot_full_timeline = True
plot_duration_curve = False
save_plot_LG = True
box_in_frame_LG = False
plot_production(df_gen_resampled=df_gen_resampled,
                df_prices_resampled=df_prices_resampled,
                DATE_START=DATE_START,
                DATE_END=DATE_END,
                interval=interval,
                fig_size=figsize,
                TITLE=title_gen,
                OUTPUT_PATH_PLOTS=OUTPUT_PATH_PLOTS,
                plot_full_timeline=plot_full_timeline,
                plot_duration_curve=plot_duration_curve,
                save_fig=save_plot_LG,
                box_in_frame=box_in_frame_LG,
                tex_font=False)




# %% Total production and load
print(f"Total production: {total_production/1e6:.2f} TWh")
load_demand = res.getDemandPerArea(area='NO')
print(f"Total load: {sum(load_demand['sum'])/1e6:.2f} TWh")



