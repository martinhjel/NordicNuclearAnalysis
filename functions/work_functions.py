from fontTools.cffLib import topDictOperators

from functions.plot_functions import *
from functions.global_functions import *
from functions.database_functions import *






##### THE FOLLOWING FUNCTIONS ARE USED AS INTERMEDIATE LINKS BETWEEN CASE_DB.PY FILES

def calcSystemCostAndMeanPriceFromDB(data: GridData, database: Database, time_SC, time_MP):
    print(f"System cost {sum(getSystemCostFromDB(data=data, db=database, timeMaxMin=time_SC).values()):.2f} EUR, or {sum(getSystemCostFromDB(data=data, db=database, timeMaxMin=time_SC).values())/1e9:.2f} Billion EUR")
    print(f"Mean area price {sum(getAreaPricesAverageFromDB(data=data, db=database, areas=None, timeMaxMin=time_MP).values()) / len(getAreaPricesAverageFromDB(data=data, db=database, areas=None, timeMaxMin=time_MP)):.2f} EUR/MWh")

    
def plot_Map(data: GridData, database: Database, time_Map, DATE_START, OUTPUT_PATH, version):
    correct_date_start = DATE_START + pd.Timedelta(hours=time_Map[0])
    correct_date_end = DATE_START + pd.Timedelta(hours=time_Map[-1])
    output_path = OUTPUT_PATH / f'prices_and_branch_utilization_map_{version}_{correct_date_start.year}_{correct_date_end.year}.html'

    create_price_and_utilization_map_FromDB(data, database, time_max_min=time_Map, output_path=output_path)



def plot_Flow_fromDB(data, db, DATE_START, time_max_min, OUTPUT_PATH_PLOTS, plot_config, chosen_connections=None):
    """
    Generates plots for AC and DC power flows.

    Parameters:
    - data: GridData object to retrieve grid data.
    - db: Database object to retrieve flow data.
    - grid_data_path: Path to the grid data.
    - time_max_min: Time range for the analysis.
    - OUTPUT_PATH_PLOTS: Directory path to save the plots.
    - by_year (bool): If True, generate separate plots for each year.
    - plot_duration_curve (bool): If True, plot duration curves instead of time series.
    - save_fig (bool): If True, save the plots as PDF files.
    """
    DATE_START = DATE_START + pd.Timedelta(hours=time_max_min[0])

    AC_interconnections, DC_interconnections = filter_connections_by_list(data, chosen_connections)
    AC_interconnections_capacity = AC_interconnections['capacity']
    DC_interconnections_capacity = DC_interconnections['capacity']

    # Get connections
    AC_dict, DC_dict = get_connections(data, chosen_connections)

    # Collect AC and DC flow data
    flow_data_AC = collect_flow_data(db, time_max_min, AC_dict, AC_interconnections_capacity, ac=True)
    flow_data_DC = collect_flow_data(db, time_max_min, DC_dict, DC_interconnections_capacity, ac=False)

    # Combine data into a single DataFrame
    flow_df = pd.concat([
        pd.DataFrame(flow_data_AC),
        pd.DataFrame(flow_data_DC)
    ], ignore_index=True)

    # Ensure OUTPUT_PATH_PLOTS is a Path object
    OUTPUT_PATH_PLOTS = pathlib.Path(OUTPUT_PATH_PLOTS)
    OUTPUT_PATH_PLOTS.mkdir(parents=True, exist_ok=True)

    if plot_config['check']:
        return flow_df
    # Plot import/ export load flow with respect to time for each interconnection
    for index, row in flow_df.iterrows():

        if plot_config['plot_by_year'] and plot_config['duration_curve']:
            # Plot duration curves for each year
            plot_duration_curve_by_year(row, DATE_START, OUTPUT_PATH_PLOTS, plot_config['save_fig'], plot_config['duration_relative'], plot_config['tex_font'])
        elif plot_config['plot_by_year'] and not plot_config['duration_curve']:
            plot_by_year(row, DATE_START, OUTPUT_PATH_PLOTS, plot_config['save_fig'], plot_config['interval'], plot_config['tex_font'])
        elif plot_config['duration_curve'] and not plot_config['plot_by_year']:
            plot_duration_curve(row, OUTPUT_PATH_PLOTS, plot_config)
        else:
            plot_time_series(row, DATE_START, OUTPUT_PATH_PLOTS, plot_config['save_fig'], plot_config['interval'], plot_config['tex_font'])


# Regular Plotting
def plot_SF_Areas_FromDB(data: GridData, database: Database, time_SF, OUTPUT_PATH_PLOTS, DATE_START, plot_config, START, END):
    """
    Plot the storage filling for given areas
    """
    storfilling = pd.DataFrame()
    for area in plot_config['areas']:
        storfilling[area] = getStorageFillingInAreaFromDB(data=data,
                                                           db=database,
                                                           areas=[area],
                                                           generator_type=['hydro', 'ror'],
                                                           relative_storage=plot_config['relative'],
                                                           timeMaxMin=time_SF)
        if plot_config['relative']:
            storfilling[area] = storfilling[area] * 100
    # Compute the correct DATE_START for this year
    start_time = datetime(START['year'], START['month'], START['day'], START['hour'], 0)
    end_time = datetime(END['year'], END['month'], END['day'], END['hour'], 0)
    storfilling.index = pd.date_range(start=start_time, end=end_time, freq='h')
    storfilling['year'] = storfilling.index.year  # Add year column to DataFrame

    # Threshold for considering reservoir empty
    empty_threshold = plot_config.get('empty_threshold', 1e-6)

    # Detect empty reservoir periods and calculate empty hours per year
    empty_summary = {}
    for area in plot_config['areas']:
        # Identify empty periods (storage <= threshold)
        is_empty = storfilling[area] <= empty_threshold
        empty_periods = storfilling[is_empty][['year']].copy()

        # Calculate total empty hours per year
        empty_hours_per_year = empty_periods.groupby('year').size()
        empty_summary[area] = {
            'empty_hours_per_year': empty_hours_per_year.to_dict(),
            'empty_periods': empty_periods.index.tolist()  # List of timestamps when empty
        }

        # Print summary for this area
        print(f"\nEmpty Reservoir Summary for Area: {area}")
        for year, hours in empty_hours_per_year.items():
            print(f"Year {year}: {hours} hours empty")
        if empty_periods.empty:
            print("No periods where reservoir was completely empty.")
    if plot_config['title'] is not None:
        title_storage_filling = f"Reservoir Filling in {'Area: ' + ', '.join(plot_config['areas'])} for period {start_time.year}-{end_time.year-1}"
    else:
        title_storage_filling = None
    plot_storage_filling_area(storfilling=storfilling,
                              DATE_START=start_time,
                              DATE_END=end_time,
                              areas=plot_config['areas'],
                              interval=plot_config['interval'],
                              title=title_storage_filling,
                              OUTPUT_PATH_PLOTS=OUTPUT_PATH_PLOTS,
                              relative=plot_config['relative'],
                              plot_by_year=plot_config['plot_by_year'],
                              save_plot=plot_config['save_fig'],
                              duration_curve=plot_config['duration_curve'],
                              tex_font=plot_config['tex_font'],
                              legend=plot_config['include_legend'],
                              fig_size=plot_config['fig_size'],
                              START=START,
                              END=END)



def plot_SF_Zones_FromDB(data: GridData, database: Database, time_SF ,OUTPUT_PATH_PLOTS, DATE_START, plot_config, START, END):
    """
    Plot the storage filling for given zones
    """
    storfilling = pd.DataFrame()
    for zone in plot_config['zones']:
        storfilling[zone] = getStorageFillingInZoneFromDB(data=data,
                                                           db=database,
                                                           zones=[zone],
                                                           generator_type=['hydro', 'ror'],
                                                           relative_storage=plot_config['relative'],
                                                           timeMaxMin=time_SF)
        if plot_config['relative']:
            storfilling[zone] = storfilling[zone] * 100
    # Compute the correct DATE_START for this year
    start_time = datetime(START['year'], START['month'], START['day'], START['hour'], 0)
    end_time = datetime(END['year'], END['month'], END['day'], END['hour'], 0)
    storfilling.index = pd.date_range(start=start_time, end=end_time, freq='h')
    storfilling['year'] = storfilling.index.year    # Add year column to DataFrame

    # Threshold for considering reservoir empty
    empty_threshold = plot_config.get('empty_threshold', 1e-6)

    # Detect empty reservoir periods and calculate empty hours per year
    empty_summary = {}
    for zone in plot_config['zones']:
        # Identify empty periods (storage <= threshold)
        is_empty = storfilling[zone] <= empty_threshold
        empty_periods = storfilling[is_empty][['year']].copy()

        # Calculate total empty hours per year
        empty_hours_per_year = empty_periods.groupby('year').size()
        empty_summary[zone] = {
            'empty_hours_per_year': empty_hours_per_year.to_dict(),
            'empty_periods': empty_periods.index.tolist()  # List of timestamps when empty
        }

        # Print summary for this area
        print(f"\nEmpty Reservoir Summary for Area: {zone}")
        for year, hours in empty_hours_per_year.items():
            print(f"Year {year}: {hours} hours empty")
        if empty_periods.empty:
            print("No periods where reservoir was completely empty.")

    if plot_config['plot_by_year'] == 1:
        for year in storfilling['year'].unique():
            if plot_config['title'] is not None:
                title_storage_filling = f"Reservoir Filling in {'Zones: ' + ', '.join(plot_config['zones'])} for year {year}"
            else:
                title_storage_filling = None
            storfilling_year = storfilling[storfilling['year'] == year]
            storfilling_year.index = pd.date_range(start_time, periods=storfilling_year.shape[0], freq='h')
            plot_storage_filling_area(storfilling=storfilling_year,
                                      DATE_START=start_time,
                                      DATE_END=end_time,
                                      areas=plot_config['zones'],
                                      interval=plot_config['interval'],
                                      title=title_storage_filling,
                                      OUTPUT_PATH_PLOTS=OUTPUT_PATH_PLOTS,
                                      relative=plot_config['relative'],
                                      plot_by_year=True,
                                      save_plot=plot_config['save_fig'],
                                      duration_curve=plot_config['duration_curve'],
                                      tex_font=plot_config['tex_font'],
                                      legend=plot_config['include_legend'],
                                      fig_size=plot_config['fig_size'],
                                      START=START,
                                      END=END)

    elif plot_config['plot_by_year'] == 2:
        if plot_config['title'] is not None:
            title_storage_filling = f"Reservoir Filling in {plot_config['zones']} for period {start_time.year}-{end_time.year}"
        else:
            title_storage_filling = None
        plot_storage_filling_area(storfilling=storfilling,
                                  DATE_START=start_time,
                                  DATE_END=end_time,
                                  areas=plot_config['zones'],
                                  interval=plot_config['interval'],
                                  title=title_storage_filling,
                                  OUTPUT_PATH_PLOTS=OUTPUT_PATH_PLOTS,
                                  relative=plot_config['relative'],
                                  plot_by_year=False,
                                  save_plot=plot_config['save_fig'],
                                  duration_curve=plot_config['duration_curve'],
                                  tex_font=plot_config['tex_font'],
                                  legend=plot_config['include_legend'],
                                  fig_size=plot_config['fig_size'],
                                  START=START,
                                  END=END)

    elif plot_config['plot_by_year'] == 3:
        if plot_config['title'] is not None:
            title_storage_filling = f"Reservoir Filling in {'Zones: ' + ', '.join(plot_config['zones'])}"
        else:
            title_storage_filling = None
        plot_storage_filling_area(storfilling=storfilling,
                                  DATE_START=start_time,
                                  DATE_END=end_time,
                                  areas=plot_config['zones'],
                                  interval=plot_config['interval'],
                                  title=title_storage_filling,
                                  OUTPUT_PATH_PLOTS=OUTPUT_PATH_PLOTS,
                                  relative=plot_config['relative'],
                                  plot_by_year=True,
                                  save_plot=plot_config['save_fig'],
                                  duration_curve=plot_config['duration_curve'],
                                  tex_font=plot_config['tex_font'],
                                  legend=plot_config['include_legend'],
                                  fig_size=plot_config['fig_size'],
                                  START=START,
                                  END=END)



def calcPlot_NP_FromDB(data: GridData, database: Database, time_NP, OUTPUT_PATH_PLOTS, DATE_START, plot_config):
    """
    Calculates and plots nodal prices from the database
    """
    nodes_in_zone = data.node[data.node['zone'] == plot_config['zone']].index.tolist() # Get all nodes in the zone
    # Get nodal prices for all nodes in the zone in one step node_prices
    node_prices = pd.DataFrame({node: getNodalPricesFromDB(database, node, time_NP) for node in nodes_in_zone})#  * EUR_MWH_TO_ORE_KWH

    correct_date_start_NP = DATE_START + pd.Timedelta(hours=time_NP[0])
    correct_date_end_NP = DATE_START + pd.Timedelta(hours=time_NP[-1])
    node_prices.index = pd.date_range(correct_date_start_NP, periods=time_NP[-1] - time_NP[0], freq='h')
    title_nodal = f"Avg. Prices in {plot_config['zone']} for period {correct_date_start_NP.year}-{correct_date_end_NP.year}"
    # title_nodal = f"Nodal Prices in {zone} for period {YEAR_START}-{YEAR_END}"
    plot_nodal_prices_FromDB(data=data,
                             node_prices=node_prices,
                             nodes_in_zone=nodes_in_zone,
                             zone=plot_config['zone'],
                             DATE_START=correct_date_start_NP,
                             DATE_END=correct_date_end_NP,
                             interval=plot_config['interval'],
                             TITLE=title_nodal,
                             plot_all_nodes=plot_config['plot_all_nodes'],
                             save_plot_nodal=plot_config['save_fig'],
                             OUTPUT_PATH_PLOTS=OUTPUT_PATH_PLOTS,
                             plot_by_year_nodal=plot_config['plot_by_year'],
                             duration_curve_nodal=plot_config['duration_curve'],
                             tex_font=plot_config['tex_font'])



def calcPlot_ZonalPrices_FromDB(data: GridData, database: Database, time_NP, OUTPUT_PATH_PLOTS, DATE_START, plot_config):
    """
    Calculates zonal prices collected from the Database.
    """
    correct_date_start_NP = DATE_START + pd.Timedelta(hours=time_NP[0])
    correct_date_end_NP = DATE_START + pd.Timedelta(hours=time_NP[-1])
    zonal_prices = pd.DataFrame()
    for zone in plot_config['zones']:
        nodes_in_zone = data.node[data.node['zone'] == zone].index.tolist() # Get all nodes in the zone
        # Get nodal prices for all nodes in the zone in one step node_prices
        node_prices = pd.DataFrame({node: getNodalPricesFromDB(database, node, time_NP) for node in nodes_in_zone})
        node_prices.index = pd.date_range(correct_date_start_NP, periods=time_NP[-1] - time_NP[0], freq='h')
        avg_node_prices = pd.DataFrame((node_prices.sum(axis=1) / len(nodes_in_zone)), columns=[f'avg_price_{zone}'])
        zonal_prices[f'avg_price_{zone}'] = avg_node_prices[f'avg_price_{zone}']

    zonal_prices.index = pd.date_range(correct_date_start_NP, periods=time_NP[-1] - time_NP[0], freq='h')
    zonal_prices[f'year'] = zonal_prices.index.year  # Add year column to DataFrame
    title_zonal = f"Avg. Prices in {'Zones: ' + ', '.join(plot_config['zones'])} for period {correct_date_start_NP.year}-{correct_date_end_NP.year}"
    plot_zonal_prices_FromDB(data=data,
                             zone_prices=zonal_prices,
                             zones=plot_config['zones'],
                             DATE_START=correct_date_start_NP,
                             DATE_END=correct_date_end_NP,
                             interval=plot_config['interval'],
                             TITLE=title_zonal,
                             save_plot_nodal=plot_config['save_fig'],
                             OUTPUT_PATH_PLOTS=OUTPUT_PATH_PLOTS,
                             plot_by_year=plot_config['plot_by_year'],
                             duration_curve_nodal=plot_config['duration_curve'],
                             tex_font=plot_config['tex_font'])


def calcPlot_HRI_FromDB(data: GridData, database: Database, time_HRI, OUTPUT_PATH_PLOTS, DATE_START, plot_config):
    """
    Calculates and plots Hydro Production, Reservoir Level and Inflow, from database
    """
    correct_date_start_HRI = DATE_START + pd.Timedelta(hours=time_HRI[0])
    correct_date_end_HRI = DATE_START + pd.Timedelta(hours=time_HRI[-1])
    df_resampled = calculate_Hydro_Res_Inflow_FromDB(data,
                                                     database,
                                                     correct_date_start_HRI,
                                                     plot_config['area'],
                                                     plot_config['genType'],
                                                     time_HRI,
                                                     plot_config['relative_storage'],
                                                     include_pump=False)
    df_resampled['year'] = df_resampled.index.year
    title_HRI = 'Hydro Production, Reservoir Filling and Inflow'
    plot_hydro_prod_res_inflow(df=df_resampled,
                               DATE_START=correct_date_start_HRI,
                               DATE_END=correct_date_end_HRI,
                               interval=plot_config['interval'],
                               TITLE=title_HRI,
                               OUTPUT_PATH_PLOTS=OUTPUT_PATH_PLOTS,
                               save_plot=plot_config['save_fig'],
                               box_in_frame=plot_config['box_in_frame'],
                               plot_full_timeline=plot_config['plot_full_timeline'],
                               tex_font=False)



def calcPlot_PLP_FromDB(data: GridData, database: Database, time_PLP, OUTPUT_PATH_PLOTS, DATE_START, plot_config):
    """
    Calculates and plots the production, load and price form database
    """
    correct_date_start_PLP = DATE_START + pd.Timedelta(hours=time_PLP[0])
    correct_date_end_PLP = DATE_START + pd.Timedelta(hours=time_PLP[-1])

    df_plp, df_plp_resampled = calc_PLP_FromDB(data, database, plot_config['area'], correct_date_start_PLP, time_PLP)
    plot_hydro_prod_demand_price(df_plp=df_plp,
                                 df_plp_resampled=df_plp_resampled,
                                 resample=plot_config['resample'],
                                 DATE_START=correct_date_start_PLP,
                                 DATE_END=correct_date_end_PLP,
                                 interval=plot_config['interval'],
                                 TITLE=plot_config['title'],
                                 save_fig=plot_config['save_fig'],
                                 plot_full_timeline=plot_config['plot_full_timeline'],
                                 box_in_frame=plot_config['box_in_frame'],
                                 OUTPUT_PATH_PLOTS=OUTPUT_PATH_PLOTS,
                                 tex_font=False)


def calcPlot_LG_FromDB(data: GridData, database: Database, time_LGT, OUTPUT_PATH_PLOTS, DATE_START, plot_config):
    """
    Calculate and plot production by type
    """
    correct_date_start_LGT = DATE_START + pd.Timedelta(hours=time_LGT[0])
    correct_date_end_LGT = DATE_START + pd.Timedelta(hours=time_LGT[-1])

    df_gen_resampled, df_prices_resampled, total_production = get_production_by_type_FromDB(data,
                                                                                            database,
                                                                                            plot_config['area'],
                                                                                            time_LGT,
                                                                                            correct_date_start_LGT)

    plot_production(df_gen_resampled=df_gen_resampled,
                    df_prices_resampled=df_prices_resampled,
                    DATE_START=correct_date_start_LGT,
                    DATE_END=correct_date_end_LGT,
                    interval=plot_config['interval'],
                    fig_size=plot_config['fig_size'],
                    TITLE=plot_config['title'],
                    OUTPUT_PATH_PLOTS=OUTPUT_PATH_PLOTS,
                    plot_full_timeline=plot_config['plot_full_timeline'],
                    plot_duration_curve=plot_config['duration_curve'],
                    save_fig=plot_config['save_fig'],
                    box_in_frame=plot_config['box_in_frame'],
                    tex_font=False)

    return df_gen_resampled, df_prices_resampled, total_production



def getProductionZonesInArea(data: GridData, db: Database, area=None, time_max_min=None, OUTPUT_PATH=None, DATE_START=None, week=None):
    """
    Fetches the production by type for all zones within a specified area and exports the result to a CSV file.
    """
    zones_in_area_prod = get_production_by_type_FromDB_ZoneLevel(data, db, area=area, time_max_min=time_max_min,
                                                                 DATE_START=DATE_START, week=week)
    zones_in_area_prod.to_csv(OUTPUT_PATH / 'data_files' / f'production_zone_level_{area}_{DATE_START.year}.csv')
    return zones_in_area_prod


def getProductionNodesInZone(data: GridData, db: Database, zone=None, time_max_min=None, OUTPUT_PATH=None,  DATE_START=None, week=None):
    """
    Retrieves the production by types for all nodes in a specified zone and exports the result to a CSV file.
    """
    nodes_in_zone_prod = get_production_by_type_FromDB_NodesInZone(data, db, zone=zone, time_max_min=time_max_min,
                                                                   DATE_START=DATE_START, week=week)
    nodes_in_zone_prod.to_csv(OUTPUT_PATH / 'data_files' / f'production_nodes_in_zone_{zone}_{DATE_START.year}.csv')
    return nodes_in_zone_prod





