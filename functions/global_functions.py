# Imports
import powergama
import pathlib
import numpy as np
import pandas as pd
import time
import folium
import math

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates

import powergama.scenarios as pgs
from IPython.display import display
import branca.colormap as cm
from powergama.GIS import _pointBetween
from powergama.database import Database  # Import Database-Class specifically
from powergama.GridData import GridData  # Import GridData-Class specifically
from functions.database_functions import *
from openpyxl import Workbook
from openpyxl.utils import get_column_letter
from datetime import datetime, timedelta


def read_grid_data(version,
                   date_start,
                   date_end,
                   data_path,
                   case,
                   ):
    """
    Reads and processes grid data for a specified year and date range.

    Parameters:
        year (int): The year for which data should be loaded.
        version (str): The version of the dataset to be used.
        date_start (str): The start date of the simulation period in 'YYYY-MM-DD' format.
        date_end (str): The end date of the simulation period in 'YYYY-MM-DD' format.
        data_path (str or pathlib.Path): The base path to the data directory.

    Returns:
        powergama.GridData: An instance of GridData containing the processed grid data.
    """
    # Calculate and print the number of simulation hours and years
    datapath_GridData = data_path / "system"
    file_storval_filling = data_path / f"storage/profiles_storval_filling_{case}_{version}.csv"
    file_30y_profiles = data_path / "timeseries_profiles.csv"

    # Initialize GridData object
    data = powergama.GridData()
    data.readGridData(nodes=datapath_GridData / f"node_{case}_{version}.csv",
                      ac_branches=datapath_GridData / f"branch_{case}_{version}.csv",
                      dc_branches=datapath_GridData / f"dcbranch_{case}_{version}.csv",
                      generators=datapath_GridData / f"generator_{case}_{version}.csv",
                      consumers=datapath_GridData / f"consumer_{case}_{version}.csv")

    # Read and process 30-year profiles
    profiles_30y = pd.read_csv(file_30y_profiles, index_col=0, parse_dates=True)
    profiles_30y["const"] = 1
    data.profiles = profiles_30y[(profiles_30y.index >= date_start) & (profiles_30y.index <= date_end)].reset_index()
    data.storagevalue_time = data.profiles[["const"]]

    # Read storage value filling data
    storval_filling = pd.read_csv(file_storval_filling)
    data.storagevalue_filling = storval_filling

    # Set the timerange and time delta for the simulation
    data.timerange = list(range(data.profiles.shape[0]))
    data.timeDelta = 1.0  # hourly data

    # Calculate and print the number of simulation hours and years
    num_hours = len(data.profiles) # data.timerange[-1] - data.timerange[0]
    print(f'Simulation hours: {num_hours}')
    num_years = num_hours / (365.2425 * 24)
    print(f'Simulation years: {np.round(num_years, 3)}')

    # Filter offshore wind farms by year:
    # data.generator = data.generator[~(data.generator["year"] > year)].reset_index(drop=True)

    # remove zero capacity generators:
    m_gen_hascap = data.generator["pmax"] > 0
    data.generator = data.generator[m_gen_hascap].reset_index(drop=True)

    return data


# Read and configure grid
def setup_grid(version,
               date_start,
               date_end,
               data_path,
               case,
               ):
    """
    Set up grid data and initialize a simulation scenario.

    This function reads grid data for the specified year and date range,
    then configures the base grid data with a specific scenario file.

    Parameters:
        year (int): The year for which data should be loaded.
        version (str): The version of the dataset to be used.
        date_start (str): The start date of the simulation period.
        date_end (str): The end date of the simulation period.

    Returns:
        data (Scenario): Configured grid data for simulation.
        time_max_min (list): List containing the start and end indices for the simulation timeframe.
    """
    print(f"Using version: {version}")
    data = read_grid_data(version, date_start, date_end, data_path, case)
    time_max_min = [0, len(data.timerange)]
    return data, time_max_min



def solve_lp(data,
             sql_file,
             loss_method,
             replace,
             nuclear_availability=None,
             week_MSO=None):
    """
    Solves a linear programming problem using the given grid data and stores the results in a SQL file.

    Parameters:
        data (powergama.GridData): The grid data to be used for the linear programming problem.
        sql_file (str): The path to the SQL file where the results will be stored.
        loss_method (str): The loss method to be used.
        replace (bool): If True, replace the existing solution with a new solution.
        nuclear_availability (float): The nuclear availability factor to be used.
        week_MSO (int): The week of MSO (Maintenance Start Order) to be used.

    Returns:
        powergama.Results: The results of the linear programming problem.
    """

    lp = powergama.LpProblem(grid=data, lossmethod=loss_method)  # lossmethod; 0=no losses, 1=linearised losses, 2=added as load
    # if replace = False, bruker kun sql_file som input
    res = powergama.Results(data, sql_file, replace=replace)
    if replace:
        start_time = time.time()
        lp.solve(res, solver="glpk", nuclear_availability=nuclear_availability, week_MSO=week_MSO)
        end_time = time.time()
        print("\nSimulation time = {:.2f} seconds".format(end_time - start_time))
        print("\nSimulation time = {:.2f} minutes".format((end_time - start_time)/60))
    return res




########################################### DATA COLLECTION ######################################################

def createZonePriceMatrix(data: GridData, database: Database, zones, year_range, TIMEZONE, SIM_YEAR_START, SIM_YEAR_END):
    zonal_price_map = pd.DataFrame(index=zones)

    for year in year_range:
        START = {"year": year, "month": 1, "day": 1, "hour": 0}
        if year >= 2020:
            END = {"year": year, "month": 12, "day": 31, "hour": 23}
        else:
            END = {"year": year + 1, "month": 1, "day": 1, "hour": 0}

        # Time setup
        try:
            start_datetime = datetime(START["year"], START["month"], START["day"], START["hour"], 0, tzinfo=TIMEZONE)
            end_datetime = datetime(END["year"], END["month"], END["day"], END["hour"], 0, tzinfo=TIMEZONE)

            time_range = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
            date_index = pd.date_range(start=start_datetime, end=end_datetime, freq='h', inclusive='left')

        except Exception as e:
            print(f"❌ Failed to generate time range for year {year}: {e}")
            continue

        for zone in zones:
            try:
                nodes_in_zone = data.node[data.node['zone'] == zone].index.tolist()
                if not nodes_in_zone:
                    print(f"⚠️ No nodes found for zone {zone} — skipping.")
                    continue

                print(f"📡 Fetching nodal prices for zone {zone} in year {year}...")

                # Fetch nodal prices for each node
                node_prices = {}
                for node in nodes_in_zone:
                    try:
                        node_prices[node] = getNodalPricesFromDB(database, node, time_range)
                    except Exception as e:
                        print(f"❌ Failed to fetch prices for node {node} in zone {zone}: {e}")
                        continue

                if not node_prices:
                    print(f"⚠️ No prices available for zone {zone} in year {year}. Skipping.")
                    continue

                df = pd.DataFrame(node_prices)
                df.index = date_index

                avg_price = df.mean(axis=1).mean()
                zonal_price_map.loc[zone, str(year)] = round(avg_price, 2)

            except Exception as e:
                print(f"❌ Failed processing zone {zone} in year {year}: {e}")
                continue

    return zonal_price_map



########################################### MAP FUNCTIONS ######################################################

def nordic_grid_map_fromDB(data, db: Database, time_range, OUTPUT_PATH, version, START, END, exchange_rate_NOK_EUR=11.38):
    """
    Generate an interactive map displaying nodal prices and branch utilization.

    This function creates a folium map that visualizes:
    - Nodes representing average nodal prices.
    - Branches representing line utilization for both AC and DC connections.

    The generated map is saved as an HTML file for easy visualization.

    Parameters:
        data (Scenario):
            The simulation data containing node and branch information.
        db (Database):
            Database object used to retrieve average prices, utilization rates, and flow data.
        time_range (list):
            List specifying the start and end time steps for the simulation.
        OUTPUT_PATH (str):
            Directory path where the HTML map file will be saved.
        version (str):
            Version identifier for the map output file.
        START (dict):
            Dictionary specifying the start date and time (e.g., {'year': 2023, 'month': 1, 'day': 1, 'hour': 0}).
        END (dict):
            Dictionary specifying the end date and time in the same format as START.
        exchange_rate_NOK_EUR (float, optional):
            Conversion rate from EUR to NOK for displaying prices in both currencies.
            Default is 11.38.

    Returns:
        None: The generated map is saved directly to the specified `OUTPUT_PATH`.
    """

    avg_nodal_prices = list(map(float, getAverageNodalPricesFromDB(db, time_range)))
    avg_area_price = {key: float(value) for key, value in getAreaPricesAverageFromDB(data, db, timeMaxMin=time_range).items()}
    avg_zone_price = getZonePricesAverageFromDB(data, db, time_range)
    ac_utilisation = list(map(float, getAverageUtilisationFromDB(data, db, time_range, branchtype="ac")))
    dc_utilisation = list(map(float, getAverageUtilisationFromDB(data, db, time_range, branchtype="dc")))
    ac_flows = convert_to_float(getAverageBranchFlowsFromDB(db, time_range, branchtype="ac"))
    dc_flows = convert_to_float(getAverageBranchFlowsFromDB(db, time_range, branchtype="dc"))

    f = folium.Figure(width=700, height=800)
    m = folium.Map(location=[data.node["lat"].mean(), data.node["lon"].mean()], zoom_start=4.4)
    m.add_to(f)

    colormap = cm.LinearColormap(['green', 'yellow', 'red'], vmin=min(avg_nodal_prices), vmax=max(avg_nodal_prices))
    colormap.caption = 'Nodal Prices'
    colormap.add_to(m)

    for i, price in enumerate(avg_nodal_prices):
        add_node_marker(data, i, price, avg_area_price, avg_zone_price, m, colormap,exchange_rate_NOK_EUR)

    line_colormap = cm.LinearColormap(['green', 'yellow', 'red'], vmin=0, vmax=1)
    line_colormap.caption = 'Branch Utilisation'
    line_colormap.add_to(m)

    add_branch_lines(data, ac_utilisation, ac_flows, 'AC', m, line_colormap)
    add_branch_lines(data, dc_utilisation, dc_flows, 'DC', m, line_colormap, dashed=True)

    start_str = f"{START['year']}_{START['month']}_{START['day']}_{START['hour']}"
    end_str = f"{END['year']}_{END['month']}_{END['day']}_{END['hour']}"
    output_path = OUTPUT_PATH / f'nordic_grid_map_{version}_{start_str}__to__{end_str}.html'
    m.save(output_path)



def nordic_grid_map(data, res, time_max_min, OUTPUT_PATH, version, exchange_rate_NOK_EUR=11.38):
    """
    Generate an interactive map of the Nordic power grid showing nodal prices, zonal prices, and line utilization.

    This function creates a folium-based map that visualizes:
    - Average nodal electricity prices using a color-coded scale.
    - Average zonal prices representing regional electricity markets.
    - National average prices (area prices) for each country.
    - Transmission line utilization for both AC and DC branches, along with flow data.

    The map offers a spatial and intuitive overview of price levels and network loading
    over a specified time period. It is saved as an HTML file for interactive exploration.

    Parameters:
        data (Scenario):
            The scenario data including node locations and network topology.
        res (ResultHandler):
            Object providing access to simulation results such as prices, utilization, and flows.
        time_max_min (list):
            A list indicating the time range to average over.
        OUTPUT_PATH (str or Path):
            Directory where the HTML file will be stored.
        version (str):
            Identifier to distinguish map output versions.
        exchange_rate_NOK_EUR (float, optional):
            Exchange rate for converting EUR to NOK when displaying prices. Default is 11.38.

    Returns:
        None: The HTML map is saved to `OUTPUT_PATH`.
    """

    avg_nodal_prices = list(map(float, res.getAverageNodalPrices(time_max_min)))
    avg_zonal_prices = res.getAverageZonalPrices(time_max_min)
    avg_area_price = {key: float(value) for key, value in res.getAreaPricesAverage(timeMaxMin=time_max_min).items()}
    ac_utilisation = list(map(float, res.getAverageUtilisation(time_max_min, branchtype="ac")))
    ac_flows = convert_to_float(res.getAverageBranchFlows(time_max_min, branchtype="ac"))
    dc_utilisation = list(map(float, res.getAverageUtilisation(time_max_min, branchtype="dc")))
    dc_flows = convert_to_float(res.getAverageBranchFlows(time_max_min, branchtype="dc"))

    f = folium.Figure(width=700, height=800)
    m = folium.Map(location=[data.node["lat"].mean(), data.node["lon"].mean()], zoom_start=4.4)
    m.add_to(f)

    colormap = cm.LinearColormap(['green', 'yellow', 'red'], vmin=min(avg_nodal_prices), vmax=max(avg_nodal_prices))
    colormap.caption = 'Nodal Prices'
    colormap.add_to(m)

    for i, price in enumerate(avg_nodal_prices):
        add_node_marker(data, i, price, avg_area_price, avg_zonal_prices, m, colormap, exchange_rate_NOK_EUR)


    line_colormap = cm.LinearColormap(['green', 'yellow', 'red'], vmin=0, vmax=1)
    line_colormap.caption = 'Branch Utilisation'
    line_colormap.add_to(m)

    add_branch_lines(data, ac_utilisation, ac_flows, 'AC', m, line_colormap)
    add_branch_lines(data, dc_utilisation, dc_flows, 'DC', m, line_colormap, dashed=True)

    output_path = OUTPUT_PATH / f'nordic_grid_map_{version}.html'
    m.save(output_path)



"""Flow Based Functions"""
def create_price_and_utilization_map(data, res, time_max_min, output_path, dc=None):
    """
        Generate a folium map displaying nodal prices and branch utilization.

        This function creates a map with nodes representing average prices and branches representing
        line utilization for the given time range. The map is saved as an HTML file.

        Parameters:
            data (Scenario):        Simulation data containing nodes and branches.
            res (Result):           Result object with nodal prices, utilization, and flows.
            time_max_min (list):    List specifying the start and end time steps for the simulation.
            output_path (str):      Path where the HTML map file will be saved.

        Returns:
            None
    """

    nodal_prices = list(map(float, res.getAverageNodalPrices(time_max_min)))
    avg_area_price = {key: float(value) for key, value in res.getAreaPricesAverage(timeMaxMin=time_max_min).items()}
    ac_utilisation = list(map(float, res.getAverageUtilisation(time_max_min, branchtype="ac")))
    ac_flows = convert_to_float(res.getAverageBranchFlows(time_max_min, branchtype="ac"))
    if dc is not None:
        dc_utilisation = list(map(float, res.getAverageUtilisation(time_max_min, branchtype="dc")))
        dc_flows = convert_to_float(res.getAverageBranchFlows(time_max_min, branchtype="dc"))

    f = folium.Figure(width=700, height=800)
    m = folium.Map(location=[data.node["lat"].mean(), data.node["lon"].mean()], zoom_start=4.4)
    m.add_to(f)

    colormap = cm.LinearColormap(['green', 'yellow', 'red'], vmin=min(nodal_prices), vmax=max(nodal_prices))
    colormap.caption = 'Nodal Prices'
    colormap.add_to(m)

    for i, price in enumerate(nodal_prices):
        add_node_marker(data, i, price, avg_area_price, m, colormap)

    line_colormap = cm.LinearColormap(['green', 'yellow', 'red'], vmin=0, vmax=1)
    line_colormap.caption = 'Branch Utilisation'
    line_colormap.add_to(m)

    add_branch_lines(data, ac_utilisation, ac_flows[2], 'AC', m, line_colormap)
    if dc is not None:
        add_branch_lines(data, dc_utilisation, dc_flows[2], 'DC', m, line_colormap, dashed=True)

    m.save(output_path)
    display(m)

def convert_to_float(data):
    """Recursively convert np.float64 values in lists to Python float"""
    if isinstance(data, list):
        return [convert_to_float(item) for item in data]
    elif isinstance(data, np.float64):
        return float(data)
    else:
        return data

def create_price_and_utilization_map_FromDB(data: GridData, db: Database, time_max_min, output_path):
    """
    Generate a folium map displaying nodal prices and branch utilization.

    This function creates a map with nodes representing average prices and branches representing
    line utilization for the given time range. The map is saved as an HTML file.

    Parameters:
        data (Scenario):        Simulation data containing nodes and branches.
        db (Database):          Result object with nodal prices, utilization, and flows.
        time_max_min (list):    List specifying the start and end time steps for the simulation.
        output_path (str):      Path where the HTML map file will be saved.
        eur_to_nok (float):     Conversion rate from EUR to NOK for price display.

    Returns:
        None
    """
    nodal_prices = list(map(float, getAverageNodalPricesFromDB(db, time_max_min)))
    avg_area_price = {key: float(value) for key, value in getAreaPricesAverageFromDB(data, db, timeMaxMin=time_max_min).items()}
    ac_utilisation = list(map(float, getAverageUtilisationFromDB(data, db, time_max_min, branchtype="ac")))
    dc_utilisation = list(map(float, getAverageUtilisationFromDB(data, db, time_max_min, branchtype="dc")))
    ac_flows = convert_to_float(getAverageBranchFlowsFromDB(db, time_max_min, branchtype="ac"))
    dc_flows = convert_to_float(getAverageBranchFlowsFromDB(db, time_max_min, branchtype="dc"))

    f = folium.Figure(width=700, height=800)
    m = folium.Map(location=[data.node["lat"].mean(), data.node["lon"].mean()], zoom_start=4.4)
    m.add_to(f)

    colormap = cm.LinearColormap(['green', 'yellow', 'red'], vmin=min(nodal_prices), vmax=max(nodal_prices))
    colormap.caption = 'Nodal Prices'
    colormap.add_to(m)

    for i, price in enumerate(nodal_prices):
        add_node_marker(data, i, price, avg_area_price, m, colormap)

    line_colormap = cm.LinearColormap(['green', 'yellow', 'red'], vmin=0, vmax=1)
    line_colormap.caption = 'Branch Utilisation'
    line_colormap.add_to(m)

    add_branch_lines(data, ac_utilisation, ac_flows[2], 'AC', m, line_colormap)
    add_branch_lines(data, dc_utilisation, dc_flows[2], 'DC', m, line_colormap, dashed=True)

    m.save(output_path)
    display(m)


def add_node_marker(data, index, price, avg_national_prices, avg_zonal_price, m, colormap, exchange_rate_NOK_EUR):
    """
    Add a marker to the map for a specific node, displaying its price and zone information.

    Parameters:
        data (Scenario):            Simulation data containing node information.
        index (int):                Node index within the data.
        price (float):              Nodal price for the given node.
        avg_national_prices (dict): Dictionary of average national prices by area.
        m (folium.Map):             Folium map object to which the marker will be added.
        colormap (LinearColormap):  Colormap for representing nodal prices.
        eur_to_nok (float):         Conversion rate from EUR to NOK. - NOT INCLUDED ANYMORE

    Returns:
        None
    """
    lat = data.node.loc[index, 'lat']
    lon = data.node.loc[index, 'lon']
    node_idx = data.node.loc[index, 'index']
    node_id = data.node.loc[index, 'id']
    node_zone = data.node.loc[index, 'zone']
    area = data.node.loc[index, 'area']
    zonal_price = avg_zonal_price.get(node_zone, 'N/A')
    area_price = avg_national_prices.get(area, 'N/A')
    nodal_price_nok = (price * exchange_rate_NOK_EUR) / 10
    zonal_price_nok = (zonal_price * exchange_rate_NOK_EUR) / 10
    area_price_nok = (area_price * exchange_rate_NOK_EUR) / 10

    # EUR_TO_ORE_PER_KWH = eur_to_nok / 1000 * 100
    # price_nok = price * EUR_TO_ORE_PER_KWH
    # area_price_nok = area_price * EUR_TO_ORE_PER_KWH if isinstance(area_price, (int, float)) else 'N/A'

    popup = folium.Popup(
        f"<b>Node index:</b> {node_idx}<br>"
        f"<b>Node id:</b> {node_id}<br>"
        f"<b>Zone:</b> {node_zone}<br>"
        f"<b>Price:</b> {price:.2f} EUR/MWh = {nodal_price_nok:.2f} Øre/KWh <br>"
        f"<b>Zonal Price:</b> {f'{zonal_price:.2f} EUR/MWh = {zonal_price_nok:.2f} Øre/KWh' if isinstance(zonal_price, (int, float)) else 'N/A'}<br>"
        f"<b>National Price:</b> {f'{area_price:.2f} EUR/MWh = {area_price_nok:.2f} Øre/KWh' if isinstance(area_price_nok, (int, float)) else 'N/A'}<br>",
        max_width=300
    )
    folium.CircleMarker(
        location=[lat, lon],
        radius=12,
        popup=popup,
        color=colormap(price),
        fill=True,
        fill_color=colormap(price),
        fill_opacity=0.7
    ).add_to(m)


def add_branch_lines(data, utilisation, flows, branch_type, m, line_colormap, dashed=False):
    """
    Add lines representing AC or DC branches on the map, showing flow and utilization.

    Parameters:
        data (Scenario):                Simulation data containing branch information.
        utilisation (list):             List of average utilization values for each branch.
        flows (list):                   List of average flow values for each branch.
        branch_type (str):              Type of branch ("AC" or "DC").
        m (folium.Map):                 Folium map object to which the lines will be added.
        line_colormap (LinearColormap): Colormap for representing utilization.
        dashed (bool):                  If True, displays lines as dashed; used for DC branches.

    Returns:
        None
    """
    branches = data.branch if branch_type == 'AC' else data.dcbranch
    for idx, row in branches.iterrows():
        utilisation_percent = utilisation[idx] * 100
        nodeA = data.node.loc[data.node['id'] == row['node_from']].iloc[0]
        nodeB = data.node.loc[data.node['id'] == row['node_to']].iloc[0]
        line_color = line_colormap(utilisation[idx])

        popup_content = folium.Popup(
            f"<b>{branch_type} Line</b><br>"
            f"<b>Power Flow:</b> {flows[2][idx]:.2f} MW<br>"
            f"<b>Utilisation:</b> {utilisation_percent:.2f}%",
            max_width=150
        )

        folium.PolyLine(
            locations=[(nodeA['lat'], nodeA['lon']), (nodeB['lat'], nodeB['lon'])],
            color=line_color,
            weight=5,
            dash_array="5, 10" if dashed else None,
            opacity=0.7,
            popup=popup_content
        ).add_to(m)

        mid_lat, mid_lon = _pointBetween((nodeA['lat'], nodeA['lon']), (nodeB['lat'], nodeB['lon']), weight=0.5)

        flow_A_to_B = flows[0][idx]  # Flyt fra A til B
        flow_B_to_A = flows[1][idx]  # Flyt fra B til A

        if flow_A_to_B >= flow_B_to_A:
            angle = math.degrees(math.atan2(nodeB['lon'] - nodeA['lon'], nodeB['lat'] - nodeA['lat']))
        else:
            angle = math.degrees(math.atan2(nodeA['lon'] - nodeB['lon'], nodeA['lat'] - nodeB['lat']))

        folium.RegularPolygonMarker(
            location=[mid_lat, mid_lon],
            fill_color=line_color,
            number_of_sides=3,
            radius=12,
            rotation=angle,
            fill_opacity=0.9,
            color=line_color,
            weight=2,
            popup="Flow direction"
        ).add_to(m)


def get_interconnections(datapath_GridData):
    """
    Retrieve and filter cross-country AC and DC branch data from CSV files.

    This function reads AC and DC branch data from CSV files, filters for
    cross-country connections, and returns separate dictionaries for each.

    Parameters:
        datapath_GridData (Path): Path to the grid data directory.

    Returns:
        tuple: Two dictionaries (AC_dict, DC_dict), where each dictionary has branch indices as keys
               and tuples of (node_from, node_to) as values for AC and DC branches, respectively.
    """
    AC_cross_country_connections, DC_cross_country_connections = filter_cross_country_connections(datapath_GridData)

    # Create dictionaries with index as key and tuple (node_from, node_to) as value for each
    AC_cross_country_dict = {
        int(row['Unnamed: 0']): (row['node_from'], row['node_to'])
        for _, row in AC_cross_country_connections.iterrows()
    }
    DC_cross_country_dict = {
        int(row['Unnamed: 0']): (row['node_from'], row['node_to'])
        for _, row in DC_cross_country_connections.iterrows()
    }
    return AC_cross_country_dict, DC_cross_country_dict



def calculate_interconnections_flow(db, datapath_GridData, time_max_min):
    """
    Calculate import, export, and average absolute flow for AC and DC interconnections.

    This function retrieves flow data for all cross-country AC and DC branches over the
    simulation period, calculates the total and average flow, and returns the results as a DataFrame.

    Parameters:
        db (Database):              Database instance containing branch flow data.
        datapath_GridData (Path):   Path to the grid data directory for retrieving branch information.
        time_max_min (list):        List containing the start and end time steps for the simulation.

    Returns:
        pd.DataFrame: DataFrame with columns [index, type, from, to, import [MWh], export [MWh],
                      average_absolute_flow [MW]], containing flow data for each interconnection.
    """
    # Get cross-country interconnections
    AC_cross_country_dict, DC_cross_country_dict = get_interconnections(datapath_GridData)

    # Combine AC and DC connections into a single dictionary with branch type for easier processing
    all_interconnections = {
        **{(idx, 'AC'): nodes for idx, nodes in AC_cross_country_dict.items()},
        **{(idx, 'DC'): nodes for idx, nodes in DC_cross_country_dict.items()}
    }
    flow_data = []
    num_time_steps = time_max_min[1]
    # Loop through each connection
    for (branch_index, branch_type), (node_from, node_to) in all_interconnections.items():
        # Get flow data for the branch
        all_branch_flows = db.getResultBranchFlow(branch_index, time_max_min, ac=(branch_type == 'AC'))

        absolute_flows = [abs(flow) for flow in all_branch_flows]
        total_absolute_flow = sum(absolute_flows)
        average_absolute_flow = total_absolute_flow / num_time_steps  # Average absolute flow

        # Calculate import and export flow based on flow direction
        import_flow = sum(flow for flow in all_branch_flows if flow < 0)  # negative flow = import
        export_flow = sum(flow for flow in all_branch_flows if flow > 0)  # positive flow = export

        # Append results to flow data list
        flow_data.append({
            'index': branch_index,
            'type': branch_type,
            'from': node_from,
            'to': node_to,
            'import [MWh]': abs(import_flow),
            'export [MWh]': abs(export_flow),
            'average_absolute_flow [MW]': average_absolute_flow
        })
    # Create DataFrame from the collected flow data
    flow_df = pd.DataFrame(flow_data, columns=['index', 'type', 'from', 'to', 'import [MWh]', 'export [MWh]',
                                               'average_absolute_flow [MW]'])
    return flow_df


def filter_cross_country_connections(grid_data_path):
    """
    Filter cross-country connections between AC and DC branches.
    """
    AC_branch_path = grid_data_path / "branch.csv"
    DC_branch_path = grid_data_path / "dcbranch.csv"
    AC_branch_df = pd.read_csv(AC_branch_path)
    DC_branch_df = pd.read_csv(DC_branch_path)
    AC_cross_country_connections = AC_branch_df[AC_branch_df['node_from'].str[:2] != AC_branch_df['node_to'].str[:2]]
    DC_cross_country_connections = DC_branch_df[DC_branch_df['node_from'].str[:2] != DC_branch_df['node_to'].str[:2]]
    return AC_cross_country_connections, DC_cross_country_connections


def filter_cross_border_connections(grid_data_path):
    """
    Filter cross-border connections between zones and countries, exclude connections within same zone.
    """
    AC_branch_path = grid_data_path / "branch.csv"
    DC_branch_path = grid_data_path / "dcbranch.csv"
    AC_branch_df = pd.read_csv(AC_branch_path)
    DC_branch_df = pd.read_csv(DC_branch_path)
    AC_cross_border_connections = AC_branch_df[AC_branch_df['node_from'].str[:3] != AC_branch_df['node_to'].str[:3]]
    DC_cross_border_connections = DC_branch_df[DC_branch_df['node_from'].str[:3] != DC_branch_df['node_to'].str[:3]]
    return AC_cross_border_connections, DC_cross_border_connections



def get_connections(datapath_GridData, chosen_connections):
    """
    Retrieve and filter AC and DC branch data from CSV files.

    This function reads AC and DC branch data from CSV files, filters for
    cross-country connections, and returns separate dictionaries for each.

    Parameters:
        datapath_GridData (Path): Path to the grid data directory.

    Returns:
        tuple: Two dictionaries (AC_dict, DC_dict), where each dictionary has branch indices as keys
               and tuples of (node_from, node_to) as values for AC and DC branches, respectively.
    """
    AC_connections, DC_connections = filter_connections_by_list(datapath_GridData, chosen_connections)

    # Create dictionaries with index as key and tuple (node_from, node_to) as value for each
    AC_dict = {
        int(row['Unnamed: 0']): (row['node_from'], row['node_to'])
        for _, row in AC_connections.iterrows()
    }
    DC_dict = {
        int(row['Unnamed: 0']): (row['node_from'], row['node_to'])
        for _, row in DC_connections.iterrows()
    }
    return AC_dict, DC_dict



def filter_connections_by_list(grid_data_path, chosen_connections=None):
    AC_branch_path = grid_data_path / "branch.csv"
    DC_branch_path = grid_data_path / "dcbranch.csv"
    AC_branch_df = pd.read_csv(AC_branch_path)
    DC_branch_df = pd.read_csv(DC_branch_path)

    if chosen_connections:
        # Filter AC branches where the pair [node_from, node_to] matches any in chosen_connections
        AC_connections = AC_branch_df[
            AC_branch_df.apply(lambda row: [row['node_from'], row['node_to']] in chosen_connections
                                           or [row['node_to'], row['node_from']] in chosen_connections, axis=1)
        ]
        # Example filter for DC branches: Difference between node_from and node_to countries
        DC_connections = DC_branch_df[
            DC_branch_df.apply(lambda row: [row['node_from'], row['node_to']] in chosen_connections
                                           or [row['node_to'], row['node_from']] in chosen_connections, axis=1)
        ]
    else:
        # If no chosen connections are provided, return all branches
        AC_connections = AC_branch_df
        DC_connections = DC_branch_df

    return AC_connections, DC_connections



def plot_LDC_interconnections(db, grid_data_path, time_max_min, OUTPUT_PATH_PLOTS, tex_font):
    AC_interconnections, DC_interconnections = filter_cross_country_connections(grid_data_path)

    AC_interconnections_capacity = AC_interconnections['capacity']
    DC_interconnections_capacity = DC_interconnections['capacity']

    # Get cross-country interconnections
    AC_cross_country_dict, DC_cross_country_dict = get_interconnections(grid_data_path)

    # AC
    flow_data_AC = []
    branch_type = "AC"
    # Loop through each connection
    for branch_index, (node_from, node_to) in AC_cross_country_dict.items():
        # Get flow data for the branch
        branch_flows = db.getResultBranchFlow(branch_index, time_max_min, ac=True)
        absolute_flows = [abs(flow) for flow in branch_flows]
        max_capacity = AC_interconnections_capacity[branch_index]

        # Append results to flow data list
        flow_data_AC.append({
            'index': branch_index,
            'type': branch_type,
            'from': node_from,
            'to': node_to,
            'load [MW]': absolute_flows,
            'capacity [MW]': max_capacity,
        })
    for item in flow_data_AC:
        item['load [MW]'] = sorted(item['load [MW]'], reverse=True)


    # DC
    flow_data_DC = []
    branch_type = "DC"
    # Loop through each connection
    for branch_index, (node_from, node_to) in DC_cross_country_dict.items():
        # Get flow data for the branch
        branch_flows = db.getResultBranchFlow(branch_index, time_max_min, ac=False)
        absolute_flows = [abs(flow) for flow in branch_flows]
        max_capacity = DC_interconnections_capacity[branch_index]

        # Append results to flow data list
        flow_data_DC.append({
            'index': branch_index,
            'type': branch_type,
            'from': node_from,
            'to': node_to,
            'load [MW]': absolute_flows,
            'capacity [MW]': max_capacity,
        })
    for item in flow_data_DC:
        item['load [MW]'] = sorted(item['load [MW]'], reverse=True)

    flow_df = pd.concat([
        pd.DataFrame(flow_data_AC),
        pd.DataFrame(flow_data_DC)
    ], ignore_index=True)

    # Plot load duration curves for each interconnection
    for index, row in flow_df.iterrows():
        plt.figure(figsize=(10, 6))  # Opprett en ny figur for hver interconnection
        plt.plot(row['load [MW]'], label=f"From: {row['from']} To: {row['to']}")
        plt.axhline(y=row['capacity [MW]'], color='red', linestyle='--', label=f"Maximum capacity: {row['capacity [MW]']:.2f} MW")
        max_y = row['capacity [MW]']
        plt.ylim(0, max_y + 50)
        plt.title(f"Load duration curve for {row['type']} connection {row['from']} <--> {row['to']}")
        plt.xlabel("Time [hours]")
        plt.ylabel("Load [MW]")
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.tight_layout()
        if tex_font:
            plt.rcParams.update({
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Computer Modern Roman"]})
        plot_file_name = OUTPUT_PATH_PLOTS / f"load_duration_curve_{row['from']}_{row['to']}.pdf"
        plt.savefig(plot_file_name)  # Save the plot as a PDF file
        plt.close()
        # plt.show()
    return flow_df









################################### ANALYSES FUNCTIONS #############################################################


# Time handling function using Python's built-in datetime objects.
def get_hour_range(YEAR_START, YEAR_END, TIMEZONE, start, end):
    """
    Beregner timeindeks for et gitt tidsintervall basert på simuleringens starttidspunkt.
    Simuleringen starter ved YEAR_START (Gitt i "General Configurations" ).
    Tar inn to tidspunkt (start og slutt) i dictionary-format og returnerer
    indeksene for disse tidspunktene i forhold til simuleringens start.

    Eksempel:
    YEAR_START: Startåret for simuleringsperioden (SQL)
    YEAR_END: Sluttåret for simuleringsperioden (SQL)
    TIMEZONE = ZoneInfo("UTC")
    START = {"year": 2005, "month": 5, "day": 5, "hour": 14}
    END = {"year": 2005, "month": 5, "day": 8, "hour": 14}
    get_hour_range(START, END)

    Funksjonen håndterer skuddår.
    Gir feilmelding dersom start og slutt ikke er innenfor perioden som SQL-filen har simulert over.
    """

    if not (YEAR_START <= start["year"] <= YEAR_END) or not (YEAR_START <= end["year"] <= YEAR_END):
        raise ValueError(f"Input years must be within {YEAR_START}-{YEAR_END}")

    start_time = datetime(YEAR_START, 1, 1, 0, 0, tzinfo=TIMEZONE)

    # Definer start- og sluttidspunktet basert på input
    start_datetime = datetime(start["year"], start["month"], start["day"], start["hour"], 0, tzinfo=TIMEZONE)
    end_datetime = datetime(end["year"], end["month"], end["day"], end["hour"], 0, tzinfo=TIMEZONE)

    # Beregn timeindeks
    start_hour_index = int((start_datetime - start_time).total_seconds() / 3600)
    end_hour_index = int((end_datetime - start_time).total_seconds() / 3600)

    print(f"Start hour index: {start_hour_index}")
    print(f"End hour index: {end_hour_index}")
    print(f"Time steps: {end_hour_index - start_hour_index}")

    return start_hour_index, end_hour_index


def auto_adjust_column_width(ws, max_width=30):
    for col_idx, col_cells in enumerate(ws.columns, 1):  # 1-based index
        lengths = []

        for cell in col_cells:
            if cell.value is not None:
                value_length = len(str(cell.value))
                lengths.append(value_length)

        if lengths:
            # Take 90th percentile length or max header length, whichever is greater
            content_width = int(np.percentile(lengths, 90))
            header_width = lengths[0]  # First cell is header
            adjusted_width = min(max(content_width, header_width) + 2, max_width)

            col_letter = get_column_letter(col_idx)
            ws.column_dimensions[col_letter].width = adjusted_width


#### Get production in specific node

def GetProductionAtSpecificNodes(Nodes, data: GridData, database: Database, start_hour, end_hour):
    """
    Henter produksjonsdata for spesifikke noder i et gitt tidsintervall.

    Args:
        Nodes (list): Liste over nodenavn (f.eks. ['NO1_1', 'NO1_2']).
        data (object): Datastruktur med informasjon om noder, generatorer osv.
        database (object): Databaseforbindelse for å hente produksjonsdata.
        start_hour (int): Startindeks for tidsserien.
        end_hour (int): Sluttindeks for tidsserien.

    Returns:
        tuple:
            - production_per_node (dict): Produksjonsdata per node og type.
            - gen_idx (list): Liste over generator-IDer per node.
            - gen_type (list): Liste over generatortyper per node.
    """
    print(f'Collecting Production at Nodes {", ".join(Nodes)}')

    # === FINN INDEKSENE FOR NODENE ===
    node_idx = [int(data.node[data.node['id'] == node].index[0]) for node in Nodes]

    # === HENT GENERATORER OG DERES TYPER ===
    gen_idx = [[gen for gen in data.getGeneratorsAtNode(idx)] for idx in node_idx]
    gen_type = [[data.generator.loc[gen, "type"] for gen in gens] for gens in gen_idx]
    flat_gen_idx = [gen for sublist in gen_idx for gen in sublist]  # Flater ut listen

    # === HENT PRODUKSJONSDATA FRA DATABASE ===
    power_output = {gen: database.getResultGeneratorPower([gen], (start_hour, end_hour)) for gen in flat_gen_idx}

    # === ORGANISERE PRODUKSJON PER NODE OG TYPE ===
    production_per_node = {node: {} for node in Nodes}
    for node, gen_list, type_list in zip(Nodes, gen_idx, gen_type):
        for gen, typ in zip(gen_list, type_list):
            production_per_node[node].setdefault(typ, []).append(power_output.get(gen, [0]))  # Setter 0 hvis data mangler

    return production_per_node, gen_idx, gen_type



def GetConsumptionAtSpecificNodes(Nodes, data: GridData, database: Database, start_hour, end_hour):
    """
    Henter forbruksdata for spesifikke noder i et gitt tidsintervall.

    Args:
        Nodes (list): Liste over nodenavn (f.eks. ['NO1_1', 'NO1_2']).
        data (object): Datastruktur med informasjon om noder.
        database (object): Databaseforbindelse for å hente forbruksdata.
        start_hour (int): Startindeks for tidsserien.
        end_hour (int): Sluttindeks for tidsserien.

    Returns:
        dict: Forbruksdata per node med kategoriene "fixed", "flex" og "sum".
    """
    print(f'Collecting Consumption at Nodes {", ".join(Nodes)}')

    # === FINN INDEKSENE FOR NODENE ===
    node_idx = [int(data.node[data.node['id'] == node].index[0]) for node in Nodes]

    # === HENT FORBRUKSDATA FOR HVER NODE ===
    consumption_per_node = {node: {} for node in Nodes}

    for node, idx in zip(Nodes, node_idx):
        area = data.node.loc[idx, "area"]  # Finn område for noden
        demand_data = getDemandPerNodeFromDB(data, database, area, node, (start_hour, end_hour))

        consumption_per_node[node]["fixed"] = demand_data["fixed"]
        consumption_per_node[node]["flex"] = demand_data["flex"]
        consumption_per_node[node]["sum"] = demand_data["sum"]

    return consumption_per_node


def GetPriceAtSpecificNodes(Nodes, data: GridData, database: Database, start_hour, end_hour):
    """
    Henter nodalpris for spesifikke noder i et gitt tidsintervall.

    Args:
        Nodes (list): Liste over nodenavn.
        data (object): Datastruktur med informasjon om noder.
        database (object): Databaseforbindelse for å hente priser.
        start_hour (int): Startindeks for tidsserien.
        end_hour (int): Sluttindeks for tidsserien.

    Returns:
        dict: Nodalpris per node.
    """
    print(f'Collecting Prices at Nodes {", ".join(Nodes)}')

    # === FINN INDEKSENE FOR NODENE ===
    node_idx = [int(data.node[data.node['id'] == node].index[0]) for node in Nodes]

    # === HENT NODALPRIS FOR HVER NODE ===
    nodal_prices = {node: database.getResultNodalPrice(idx, (start_hour, end_hour)) for node, idx in zip(Nodes, node_idx)}

    return nodal_prices




def ExportToExcel(Nodes, production_per_node, consumption_per_node, nodal_prices_per_node, reservoir_filling_per_node, storage_cap, flow_data, START, END, case, version, OUTPUT_PATH):
    """
    Eksporterer produksjons-, forbruks-, fyllingsgrads- og nodalprisdata til en Excel-fil.

    Args:
        Nodes (list): Liste over nodenavn.
        production_per_node (dict): Produksjonsdata per node og type.
        consumption_per_node (dict): Forbruksdata per node.
        nodal_prices_per_node (dict): Nodalpriser per node.
        reservoir_filling_per_node (dict): Reservoarfylling per node.
        flow_data (DataFrame): Flow data for valgte linjer
        START (dict): Starttidspunkt som dictionary (f.eks. {"year": 2019, "month": 5, "day": 1, "hour": 12}).
        END (dict): Sluttidspunkt som dictionary (f.eks. {"year": 2019, "month": 6, "day": 1, "hour": 12}).
        case (str): Navn på caset.
        version (str): Versjonsnummer.
        OUTPUT_PATH: Filsti for lagring.

    Returns:
        str: Filnavn på den lagrede Excel-filen.
    """
    print(f'Collecting Data to Excel')

    # Konverter START og END til datetime-objekter
    start_datetime = datetime(START["year"], START["month"], START["day"], START["hour"])
    end_datetime = datetime(END["year"], END["month"], END["day"], END["hour"])

    all_types = ["nuclear", "hydro", "biomass", "ror", "wind_on", "wind_off", "solar", "fossile_other", "fossile_gas"]

    # === GENERER TIDSSTEG BASERT PÅ get_hour_range() ===
    # time_stamps = [start_datetime + timedelta(hours=i) for i in range(int((end_datetime - start_datetime).total_seconds() // 3600) + 1)]
    datetime_range = pd.date_range(start=start_datetime, end=end_datetime, freq='h', inclusive='left')


    # === GENERER FILNAVN ===
    # timestamp = datetime.now().strftime("%Y-%m-%d")
    start_str = start_datetime.strftime("%Y-%m-%d-%H")
    end_str = end_datetime.strftime("%Y-%m-%d-%H")
    filename = f"Prod_demand_nodes_{case}_{version}_{start_str}_to_{end_str}.xlsx"

    # === OPPRETT NY WORKBOOK ===
    wb = Workbook()

    for node in Nodes:
        # === OPPRETT ARKFANER FOR HVER NODE ===
        ws_production = wb.create_sheet(f"Production {node}")
        ws_consumption = wb.create_sheet(f"Consumption {node}")
        ws_price = wb.create_sheet(f"Price {node}")
        ws_reservoir = wb.create_sheet(f"Reservoir {node}")

        # === LEGG TIL OVERSKRIFTER ===
        ws_production.append(["Timestamp"] + all_types)
        ws_consumption.append(["Timestamp", "Fixed", "Flexible", "Consumption"])
        ws_price.append(["Timestamp", "Nodal Price"])
        ws_reservoir.append(["Timestamp", "Reservoir Filling", "Max storage capacity", "Reservoir Filling [%]"])


        # === FYLL PRODUKSJONSARKET ===
        for t, timestamp in enumerate(datetime_range):
            row = [timestamp.strftime("%Y-%m-%d %H:%M")]  # Formater tid riktig
            for typ in all_types:
                values = production_per_node[node].get(typ, [[0]])[0]  # Fjern dobbel liste-nesting
                value = values[t] if t < len(values) else 0  # Hent riktig indeks eller sett 0
                row.append(value)
            ws_production.append(row)
        auto_adjust_column_width(ws_production)

        # === FYLL FORBRUKSARKET ===
        for t, timestamp in enumerate(datetime_range):
            fixed = consumption_per_node[node]["fixed"]
            flex = consumption_per_node[node]["flex"]
            total = consumption_per_node[node]["sum"]

            ws_consumption.append([
                timestamp.strftime("%Y-%m-%d %H:%M"),
                fixed[t] if t < len(fixed) else 0,
                flex[t] if t < len(flex) else 0,
                total[t] if t < len(total) else 0,
            ])
        auto_adjust_column_width(ws_consumption)

        # === FYLL NODALPRISARKET ===
        for t, timestamp in enumerate(datetime_range):
            nodal_price = nodal_prices_per_node[node]  # Hent liste over nodalpriser for noden
            price_value = nodal_price[t] if t < len(nodal_price) else 0  # Hent pris eller sett 0 hvis ikke nok data

            ws_price.append([
                timestamp.strftime("%Y-%m-%d %H:%M"),
                price_value,
            ])
        auto_adjust_column_width(ws_price)

        # === FYLL RESERVOARFYLLINGSARKET ===
        for t, timestamp in enumerate(datetime_range):
            # Henter og pakker ut reservoardata
            if node in reservoir_filling_per_node:
                reservoir_data = reservoir_filling_per_node.get(node, [])
                if isinstance(reservoir_data, list):  # Sikrer riktig format
                    reservoir_values = [sum(x) for x in zip(*reservoir_data)]  # Slår sammen flere generatorer
                else:
                    reservoir_values = [0] * len(datetime_range)  # Hvis data er feil format
            else:
                reservoir_values = [0] * len(datetime_range)  # Hvis noden ikke finnes

            # Hent riktig verdi fra tidsserien eller sett 0
            reservoir_value = reservoir_values[t] if t < len(reservoir_values) else 0

            # Hent maks kapasitet for noden
            max_capacity = storage_cap.get(node, 0)  # Standard 0 hvis ikke funnet

            # Beregn fyllingsgrad (%)
            filling_percentage = (reservoir_value / max_capacity) * 100 if max_capacity > 0 else 0

            ws_reservoir.append([
                timestamp.strftime("%Y-%m-%d %H:%M"),
                round(reservoir_value, 4),  # Rund av til 4 desimaler
                round(max_capacity, 4) if max_capacity > 0 else "",  # Tom celle hvis ikke kapasitet
                round(filling_percentage, 8)  # Rund av til 8 desimaler
            ])
        auto_adjust_column_width(ws_reservoir)

    # === LEGG TIL FLOW-DATA HVIS TILGJENGELIG ===
    if flow_data is not None:
        for idx, row in flow_data.iterrows():
            from_node = row['from']
            to_node = row['to']
            sheet_name = f"Flow {from_node} → {to_node}"[:31]  # Excel begrensning

            # Forsøk å laste liste
            try:
                load_list = eval(row['load [MW]']) if isinstance(row['load [MW]'], str) else row['load [MW]']
            except Exception as e:
                print(f"❌ Hopper over {from_node} → {to_node}: feil i lesing av data ({e})")
                continue

            if len(load_list) != len(datetime_range):
                print(
                    f"⚠️ Hopper over {from_node} → {to_node}: Lengdemismatch ({len(load_list)} vs {len(datetime_range)})")
                continue

            if sheet_name in wb.sheetnames:
                print(f"⚠️ Ark {sheet_name} finnes allerede – hopper over.")
                continue

            # Lag ark og legg inn data
            ws_flow = wb.create_sheet(sheet_name)
            ws_flow.append(["Timestamp", "Load [MW]"])

            for t, timestamp in enumerate(datetime_range):
                ws_flow.append([
                    timestamp.strftime("%Y-%m-%d %H:%M"),
                    load_list[t]
                ])
            auto_adjust_column_width(ws_flow)

    # Fjern default ark
    if "Sheet" in wb.sheetnames:
        wb.remove(wb["Sheet"])

    # === LAGRE FIL ===
    filepath = pathlib.Path(OUTPUT_PATH) / filename
    wb.save(filepath)

    print(f"\nExcel-fil '{filename}' er lagret i {OUTPUT_PATH}!")

    return filename


def GetReservoirFillingAtSpecificNodes(Nodes, data: GridData, database: Database, start_hour, end_hour):
    """
    Henter reservoarfylling og maksimal kapasitet for spesifikke noder.
    """
    print(f'Collecting Reservoir Filling at Nodes {", ".join(Nodes)}')

    # === FINN INDEKSENE FOR NODENE ===
    node_idx = [int(data.node[data.node['id'] == node].index[0]) for node in Nodes]

    # === HENT LAGRINGSENHETER OG DERES KAPASITET ===
    storage_data = data.generator[
        (data.generator["node"].isin(Nodes)) &
        (data.generator["storage_cap"] > 0) &
        (data.generator["type"] == "hydro")
    ][["node", "storage_cap"]]

    storage_idx = storage_data.groupby("node").apply(lambda x: list(x.index)).to_dict()
    storage_cap = storage_data.groupby("node")["storage_cap"].sum().to_dict()  # Summerer kapasitet for hver node

    flat_storage_idx = [gen for sublist in storage_idx.values() for gen in sublist]

    # === HENT RESERVOARFYLLINGSNIVÅ FRA DATABASE ===
    storage_filling = {gen: database.getResultStorageFilling(gen, (start_hour, end_hour)) for gen in flat_storage_idx}

    # === ORGANISERE DATA ===
    reservoir_filling_per_node = {node: [] for node in Nodes}

    for node, gen_list in storage_idx.items():
        node_values = []
        for gen in gen_list:
            node_values.append(storage_filling.get(gen, [0]))  # Hent fyllingsdata eller 0
        reservoir_filling_per_node[node] = node_values

    return reservoir_filling_per_node, storage_cap



# === FLOW TO EXCEL ===

def getFlowDataOnBranches(db: Database, time_max_min, grid_data_path, chosen_connections):
    """
    Collect flow on chosen connections.
    :param db:
    :param time_max_min:
    :param grid_data_path:
    :param chosen_connections:
    :return: flow_df
    """
    print(f'Collecting Flow Data at Lines {", ".join([f"{f} → {t}" for f, t in chosen_connections])}')


    AC_interconnections, DC_interconnections = filter_connections_by_list(grid_data_path, chosen_connections)
    AC_interconnections_capacity = AC_interconnections['capacity']
    DC_interconnections_capacity = DC_interconnections['capacity']

    # Get connections
    AC_dict, DC_dict = get_connections(grid_data_path, chosen_connections)

    # Collect AC and DC flow data
    flow_data_AC = collect_flow_data(db, time_max_min, AC_dict, AC_interconnections_capacity, ac=True)
    flow_data_DC = collect_flow_data(db, time_max_min, DC_dict, DC_interconnections_capacity, ac=False)

    # Combine data into a single DataFrame
    flow_df = pd.concat([
        pd.DataFrame(flow_data_AC),
        pd.DataFrame(flow_data_DC)
    ], ignore_index=True)

    return flow_df


def writeFlowToExcel(flow_data, START, END, OUTPUT_PATH, case, version):
    """
    Writes flow data to Excel spreadsheet
    Args:
        flow_data (pandas.DataFrame): flow data from getFlowDataOnBranches
        START (pandas.DataFrame): start time of flow data
        END (pandas.DataFrame): end time of flow data
        TIMEZONE (pandas.DataFrame): timezone of flow data
        OUTPUT_PATH (pathlib.Path): output path

    Returns:
        str: Excel spreadsheet path
    """
    # Konverter START og END til datetime-objekter
    start_datetime = datetime(START["year"], START["month"], START["day"], START["hour"])
    end_datetime = datetime(END["year"], END["month"], END["day"], END["hour"])

    datetime_range = pd.date_range(start=start_datetime, end=end_datetime, freq='h', inclusive='left')

    # Create filename with timestamp
    filename = f"flow_data_{case}_{version}.xlsx"
    full_path = OUTPUT_PATH / filename

    # Track if any sheet was written
    sheet_written = False

    with pd.ExcelWriter(full_path, engine='openpyxl') as writer:
        for idx, row in flow_data.iterrows():
            from_node = row['from']
            to_node = row['to']
            sheet_name = f"{from_node} → {to_node}"[:31]

            # Load list safely
            try:
                load_list = eval(row['load [MW]']) if isinstance(row['load [MW]'], str) else row['load [MW]']
            except Exception as e:
                print(f"❌ Failed to parse load for {from_node} → {to_node}: {e}")
                continue

            # Length check
            if len(load_list) != len(datetime_range):
                print(
                    f"⚠️ Skipping {from_node} → {to_node}: Length mismatch ({len(load_list)} vs {len(datetime_range)})")
                continue

            # Write sheet
            flow_df = pd.DataFrame({
                "Datetime": datetime_range,
                "Load [MW]": load_list
            })
            flow_df.to_excel(writer, sheet_name=sheet_name, index=False)
            sheet_written = True
            worksheet = writer.sheets[sheet_name]

            # Auto-adjust column widths
            for idx, col in enumerate(flow_df.columns, 1):
                max_length = max(flow_df[col].astype(str).map(len).max(), len(col)) + 2
                worksheet.column_dimensions[get_column_letter(idx)].width = max_length

    if not sheet_written:
        print("❗️No valid sheets were written. Excel file was not saved.")
        return None

    print(f"\n✅ Excel file saved at: {full_path}")
    return str(full_path)
