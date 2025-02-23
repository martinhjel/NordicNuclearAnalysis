from functions.global_functions import *
from functions.database_functions import *


def calcSystemCostAndMeanPriceFromDB(data: GridData, database: Database, time_SC, time_MP):
    print(f"System cost {sum(getSystemCostFromDB(data=data, db=database, timeMaxMin=time_SC).values()):.2f} EUR, or {sum(getSystemCostFromDB(data=data, db=database, timeMaxMin=time_SC).values())/1e9:.2f} Billion EUR")
    print(f"Mean area price {sum(getAreaPricesAverageFromDB(data=data, db=database, areas=None, timeMaxMin=time_MP).values()) / len(getAreaPricesAverageFromDB(data=data, db=database, areas=None, timeMaxMin=time_MP)):.2f} EUR/MWh")



def plot_Map(data: GridData, database: Database, time_Map, OUTPUT_PATH, version):
    output_path = OUTPUT_PATH / f'prices_and_branch_utilization_map_{version}.html'
    create_price_and_utilization_map_FromDB(data, database, time_max_min=time_Map, output_path=output_path)




def plot_SF_Areas_FromDB(data: GridData, database: Database, time_SF, OUTPUT_PATH_PLOTS, DATE_START, plot_config):
    storfilling = pd.DataFrame()

    for area in plot_config['areas']:
        storfilling[area] = getStorageFillingInAreasFromDB(data=data,
                                                           db=database,
                                                           areas=[area],
                                                           generator_type="hydro",
                                                           relative_storage=plot_config['relative'],
                                                           timeMaxMin=time_SF)
        if plot_config['relative']:
            storfilling[area] = storfilling[area] * 100

    # Compute the correct DATE_START for this year
    correct_date_start_SF = DATE_START + pd.Timedelta(hours=time_SF[0])
    correct_date_end_SF = DATE_START + pd.Timedelta(hours=time_SF[-1])

    storfilling.index = pd.date_range(correct_date_start_SF, periods=time_SF[-1] - time_SF[0], freq='h')
    storfilling['year'] = storfilling.index.year  # Add year column to DataFrame
    title_storage_filling = f"Reservoir Filling in {plot_config['areas']} for period {correct_date_start_SF.year}-{correct_date_end_SF.year}"
    plot_storage_filling_area(storfilling=storfilling,
                              DATE_START=correct_date_start_SF,
                              DATE_END=correct_date_end_SF,
                              areas=plot_config['areas'],
                              interval=plot_config['interval'],
                              title=title_storage_filling,
                              OUTPUT_PATH_PLOTS=OUTPUT_PATH_PLOTS,
                              relative=plot_config['relative'],
                              plot_by_year=plot_config['plot_by_year'],
                              save_plot=plot_config['save_fig'],
                              duration_curve=plot_config['duration_curve'],
                              tex_font=False)



def plot_SF_Zones_FromDB(data: GridData, database: Database, time_SF ,OUTPUT_PATH_PLOTS, DATE_START, plot_config):
    storfilling = pd.DataFrame()
    # TODO: Fix plot by year til å faktisk plotte flere plots for hvert år.

    for zone in plot_config['zones']:
        storfilling[zone] = getStorageFillingInZonesFromDB(data=data,
                                                           db=database,
                                                           zones=[zone],
                                                           generator_type="hydro",
                                                           relative_storage=plot_config['relative'],
                                                           timeMaxMin=time_SF)
        if plot_config['relative']:
            storfilling[zone] = storfilling[zone] * 100

    # Compute the correct DATE_START for this year
    correct_date_start_SF = DATE_START + pd.Timedelta(hours=time_SF[0])
    correct_date_end_SF = DATE_START + pd.Timedelta(hours=time_SF[-1])

    storfilling.index = pd.date_range(correct_date_start_SF, periods=time_SF[-1] - time_SF[0], freq='h')
    storfilling['year'] = storfilling.index.year    # Add year column to DataFrame

    if plot_by_year:
        for year in storfilling['year'].unique():
            title_storage_filling = f"Reservoir Filling in {'Zones: ' + ', '.join(plot_config['zones'])} for year {year}"
            storfilling_year = storfilling[storfilling['year'] == year]
            storfilling_year.index = pd.date_range(correct_date_start_SF, periods=storfilling_year.shape[0], freq='h')
            plot_storage_filling_area(storfilling=storfilling_year,
                                      DATE_START=correct_date_start_SF,
                                      DATE_END=correct_date_end_SF,
                                      areas=plot_config['zones'],
                                      interval=plot_config['interval'],
                                      title=title_storage_filling,
                                      OUTPUT_PATH_PLOTS=OUTPUT_PATH_PLOTS,
                                      relative=plot_config['relative'],
                                      plot_by_year=plot_config['plot_by_year'],
                                      save_plot=plot_config['save_fig'],
                                      duration_curve=plot_config['duration_curve'],
                                      tex_font=False)

    else:
        title_storage_filling = f"Reservoir Filling in {plot_config['zones']} for period {correct_date_start_SF.year}-{correct_date_end_SF.year}"
        plot_storage_filling_area(storfilling=storfilling,
                                  DATE_START=correct_date_start_SF,
                                  DATE_END=correct_date_end_SF,
                                  areas=plot_config['zones'],
                                  interval=plot_config['interval'],
                                  title=title_storage_filling,
                                  OUTPUT_PATH_PLOTS=OUTPUT_PATH_PLOTS,
                                  relative=plot_config['relative'],
                                  plot_by_year=plot_config['plot_by_year'],
                                  save_plot=plot_config['save_fig'],
                                  duration_curve=plot_config['duration_curve'],
                                  tex_font=False)



def calcPlot_NP_FromDB(data: GridData, database: Database, time_NP, OUTPUT_PATH_PLOTS, DATE_START, plot_config):

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
                             tex_font=False)




def calcPlot_HRI_FromDB(data: GridData, database: Database, time_HRI, OUTPUT_PATH_PLOTS, DATE_START, plot_config):


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




def getProductionZonesInArea(data: GridData, db: Database, area=None, time_max_min=None, DATE_START=None, week=None):
    """
    Fetches the production by type for all zones within a specified area and exports the result to a CSV file.
    """
    zones_in_area_prod = get_production_by_type_FromDB_ZoneLevel(data, db, area=area, time_max_min=time_max_min,
                                                                 DATE_START=DATE_START, week=week)
    zones_in_area_prod.to_csv(f'production_zone_level_{area}_{DATE_START.year}.csv')
    return zones_in_area_prod


def getProductionNodesInZone(data: GridData, db: Database, zone=None, time_max_min=None, DATE_START=None, week=None):
    """
    Retrieves the production by types for all nodes in a specified zone and exports the result to a CSV file.
    """
    nodes_in_zone_prod = get_production_by_type_FromDB_NodesInZone(data, db, zone=zone, time_max_min=time_max_min,
                                                                   DATE_START=DATE_START, week=week)
    nodes_in_zone_prod.to_csv(f'production_nodes_in_zone_{zone}_{DATE_START.year}.csv')
    return nodes_in_zone_prod