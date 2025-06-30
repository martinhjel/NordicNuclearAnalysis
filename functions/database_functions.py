"""
This file contains functions that are used to interact with the database.
"""
import csv
import numpy as np
import pandas as pd
from collections import defaultdict
from powergama.database import Database  # Import Database-Class specifically
from powergama.GridData import GridData

def getSystemCostFromDB(data: GridData, db: Database, timeMaxMin):
    """
    This function calculates the system cost from the database.

    Parameters
    ----------
    data : dict
        The data dictionary.
    db : Database
        The database object.
    timeMaxMin : list (default = None)
            [min, max] - lower and upper time interval

    Returns
    -------
    systemcost : dict
        The system cost.
    """
    generation_per_gen = db.getResultGeneratorPowerSum(timeMaxMin)
    fuelcost_per_gen = data.generator["fuelcost"]
    areas_per_gen = [
        data.node["area"][data.node["id"] == n].tolist()[0] for n in data.generator["node"]
    ]
    allareas = data.getAllAreas()
    generationcost = dict()
    for a in allareas:
        generationcost[a] = sum(
            [
                generation_per_gen[i] * fuelcost_per_gen[i]
                for i in range(len(areas_per_gen))
                if areas_per_gen[i] == a
            ]
        )
    return generationcost


def getZonePricesAverageFromDB(data: GridData, database: Database, time_range):
    """
    Calculate average zonal prices based on nodal prices.

    This function retrieves nodal prices from the database, associates them with their respective zones,
    and calculates the average price for each zone. The results are returned as a dictionary.

    Parameters:
        data (GridData):
            Simulation data containing node information, including zones.
        database (Database):
            Database object used to retrieve nodal prices.
        time_range (list):
            List specifying the time range for which data should be retrieved.

    Returns:
        dict: A dictionary where keys are zone names and values are their corresponding average prices.
    """
    avg_nodal_prices = list(map(float, getAverageNodalPricesFromDB(database, time_range)))
    zones = data.node['zone']
    combined = pd.concat([zones, pd.Series(avg_nodal_prices)], axis=1)
    combined.columns = ['zone', 'price']
    avg_zonal_prices = combined.groupby('zone')['price'].mean().to_dict()

    return avg_zonal_prices




def getAverageNodalPricesFromDB(db: Database, timeMaxMin):
    """
    Average nodal price over a given time period

    Parameters
    ----------
    db : Database
        The database object.
    timeMaxMin : list (default = None)
        [min, max] - lower and upper time interval

    Returns
    -------
    avg_prices : list
        The average prices.
    """

    avg_prices = db.getResultNodalPricesMean(timeMaxMin)
    # use asarray to convert None to nan
    avg_prices = np.asarray(avg_prices, dtype=float)
    return avg_prices


def getNodalPricesFromDB(db: Database, node, timeMaxMin = None):
    """
    Get nodal prices for a given node over a given time period

    Parameters
    ----------
    db : Database
        The database object.
    node : int
        The node.
    timeMaxMin : list (default = None)
        [min, max] - lower and upper time interval

    Returns
    -------
    prices : list
        The prices.
    """
    prices = db.getResultNodalPrice(node, timeMaxMin)
    # use asarray to convert None to nan
    prices = np.asarray(prices, dtype=float)
    return prices


def getZonePricesAverageFromDB(data: GridData, database: Database, time_range):
    """
    Calculate average zonal prices based on nodal prices.

    This function retrieves nodal prices from the database, associates them with their respective zones,
    and calculates the average price for each zone. The results are returned as a dictionary.

    Parameters:
        data (GridData):
            Simulation data containing node information, including zones.
        database (Database):
            Database object used to retrieve nodal prices.
        time_range (list):
            List specifying the time range for which data should be retrieved.

    Returns:
        dict: A dictionary where keys are zone names and values are their corresponding average prices.
    """
    avg_nodal_prices = list(map(float, getAverageNodalPricesFromDB(database, time_range)))
    zones = data.node['zone']
    combined = pd.concat([zones, pd.Series(avg_nodal_prices)], axis=1)
    combined.columns = ['zone', 'price']
    avg_zonal_prices = combined.groupby('zone')['price'].mean().to_dict()

    return avg_zonal_prices


def getAreaPricesAverageFromDB(data: GridData, db: Database, areas=None, timeMaxMin=None):
    """
    Time average of weighted average nodal price per area

    Parameters
    ----------
    data : dict
        The data dictionary.
    db : Database
        The database object.
    areas : list (default = None)
        List of areas.
    timeMaxMin : list (default = None)
        [min, max] - lower and upper time interval

    Returns
    -------
    avg_area_price : dict
        The average area price.
    """

    if areas is None:
        areas = data.getAllAreas()

    avg_nodal_prices = getAverageNodalPricesFromDB(db, timeMaxMin)
    all_loads = data.getConsumersPerArea()
    avg_area_price = {}

    for area in areas:
        nodes_in_area = [i for i, n in enumerate(data.node.area) if n == area]
        node_weight = [0] * len(data.node.id)
        if area in all_loads:
            loads = all_loads[area]
            for ld in loads:
                the_node = data.consumer.node[ld]
                the_load = data.consumer.demand_avg[ld]
                node_idx = data.node.id.tolist().index(the_node)
                node_weight[node_idx] += the_load
            sumWeight = sum(node_weight)
            node_weight = [a / sumWeight for a in node_weight]

            prices = [node_weight[i] * avg_nodal_prices[i] for i in nodes_in_area]
        else:
            # flat weight if there are no loads in area
            prices = [avg_nodal_prices[i] for i in nodes_in_area]
        avg_area_price[area] = sum(prices)
    return avg_area_price


# def getStorageFillingInAreasFromDB(data: GridData, db: Database, areas, generator_type, relative_storage, timeMaxMin):
#     """
#     Get the storage filling in areas from the database.
#
#     Parameters
#     ----------
#     data : dict
#         The data dictionary.
#     db : Database
#         The database object.
#     areas : list
#         List of areas.
#     generator_type : str
#         The generator type.
#     relative_storage : bool
#         If True, the relative storage is returned.
#     timeMaxMin : list (default = None)
#         [min, max] - lower and upper time interval
#
#     Returns
#     -------
#     filling : dict
#         The storage filling.
#     """
#     storageGen = data.getIdxGeneratorsWithStorage()
#     storageTypes = data.generator.type
#     nodeNames = data.generator.node
#     nodeAreas = data.node.area
#     storCapacities = data.generator.storage_cap
#     generators = []
#     capacity = 0
#     for gen in storageGen:
#         area = nodeAreas[data.node.id.tolist().index(nodeNames[gen])]
#         if area in areas and storageTypes[gen] == generator_type:
#             generators.append(gen)
#             if relative_storage:
#                 capacity += storCapacities[gen]
#         filling = db.getResultStorageFillingMultiple(generators, timeMaxMin, capacity)
#     return filling

def getStorageFillingInAreaFromDB(data: GridData, db: Database, areas, generator_type, relative_storage, timeMaxMin):
    """
    Get the storage filling in areas from the database using a single query.

    Parameters
    ----------
    data : GridData
        The data object containing generator and node information.
    db : Database
        The database object.
    areas : list
        List of areas.
    generator_type : str
        The generator type.
    relative_storage : bool
        If True, the relative storage is returned.
    timeMaxMin : list
        [min, max] - lower and upper time interval.

    Returns
    -------
    filling : dict
        The storage filling, aggregated by timestep.
    """
    # Get storage generators and their properties
    storageGen = data.getIdxGeneratorsWithStorage()
    storageTypes = data.generator.type
    nodeNames = data.generator.node
    nodeAreas = data.node.area
    storCapacities = data.generator.storage_cap

    # Filter generators by area and type
    generators = []
    total_capacity = 0
    for gen in storageGen:
        area = nodeAreas[data.node.id.tolist().index(nodeNames[gen])]
        if area in areas and storageTypes[gen] in generator_type:
            generators.append(gen)
            if relative_storage:
                total_capacity += storCapacities[gen]

    if not generators:
        return {}  # Return empty dict if no generators match

    # Query the database
    rows = db.getStorageFillingForGenerators(generators, timeMaxMin)

    # Process results
    filling = {}
    for row in rows:
        timestep, storage = row
        value = storage / total_capacity if relative_storage and total_capacity > 0 else storage
        filling[timestep] = value

    return filling


# def getStorageFillingInZoneFromDB(data: GridData, db: Database, zones, generator_type, relative_storage, timeMaxMin):
#     """
#     Get the storage filling in areas from the database.
#
#     Parameters
#     ----------
#     data : dict
#         The data dictionary.
#     db : Database
#         The database object.
#     zones : list
#         List of zones.
#     generator_type : str
#         The generator type.
#     relative_storage : bool
#         If True, the relative storage is returned.
#     timeMaxMin : list (default = None)
#         [min, max] - lower and upper time interval
#
#     Returns
#     -------
#     filling : dict
#         The storage filling.
#     """
#     storageGen = data.getIdxGeneratorsWithStorage()
#     storageTypes = data.generator.type
#     nodeNames = data.generator.node
#     nodeZones = data.node.zone
#     storCapacities = data.generator.storage_cap
#     generators = []
#     capacity = 0
#     for gen in storageGen:
#         zone = nodeZones[data.node.id.tolist().index(nodeNames[gen])]
#         if zone in zones and storageTypes[gen] == generator_type:
#             generators.append(gen)
#             if relative_storage:
#                 capacity += storCapacities[gen]
#         filling = db.getResultStorageFillingMultiple(generators, timeMaxMin, capacity)
#     return filling

def getStorageFillingInZoneFromDB(data: GridData, db: Database, zones, generator_type, relative_storage, timeMaxMin):
    """
    Get the storage filling in areas from the database using a single query.

    Parameters
    ----------
    data : GridData
        The data object containing generator and node information.
    db : Database
        The database object.
    zones : list
        List of areas.
    generator_type : str
        The generator type.
    relative_storage : bool
        If True, the relative storage is returned.
    timeMaxMin : list
        [min, max] - lower and upper time interval.

    Returns
    -------
    filling : dict
        The storage filling, aggregated by timestep.
    """
    # Get storage generators and their properties
    storageGen = data.getIdxGeneratorsWithStorage()
    storageTypes = data.generator.type
    nodeNames = data.generator.node
    nodeZones = data.node.zone
    storCapacities = data.generator.storage_cap

    # Filter generators by area and type
    generators = []
    total_capacity = 0
    for gen in storageGen:
        zone = nodeZones[data.node.id.tolist().index(nodeNames[gen])]
        if zone in zones and storageTypes[gen] in generator_type:
            generators.append(gen)
            if relative_storage:
                total_capacity += storCapacities[gen]

    if not generators:
        return {}  # Return empty dict if no generators match

    # Query the database
    rows = db.getStorageFillingForGenerators(generators, timeMaxMin)

    # Process results
    filling = {}
    for row in rows:
        timestep, storage = row
        value = storage / total_capacity if relative_storage and total_capacity > 0 else storage
        filling[timestep] = value

    return filling


def getProductionPerAreaFromDB(data: GridData, database, time_Prod, area):

    # === FINN INDEKSENE FOR NODENE I GITT OMRÅDE ===
    node_idx = data.node[data.node['id'].str.startswith(area)].index.tolist()

    # === HENT ALLE GENERATORER VED DISSE NODENE ===
    gen_idx = [[gen for gen in data.getGeneratorsAtNode(idx)] for idx in node_idx]
    flat_gen_idx = [gen for sublist in gen_idx for gen in sublist]  # Flater ut listen

    # === SUMMER PRODUKSJON ===
    # totalProd = 0
    # for gen in flat_gen_idx:
    #     prod = database.getResultGeneratorPower([gen], time_Prod)
    #     totalProd += sum(prod)

    totalProd = sum(database.getResultGeneratorPower(flat_gen_idx, time_Prod))

    return totalProd



def getProductionForAllNodesFromDB(data: GridData, database: Database, time_Prod):
    """
    Returns total production per node over the given time range.

    Parameters
    ----------
    data : GridData
        Contains node list & generator‑to‑node mapping.
    database : Database
        Has getResultGeneratorPower(gen_ids, time_range) → List[List[float]]
    time_Prod : list
        [min, max] time window for which to fetch generation.

    Returns
    -------
    prod_per_node : dict[int, float]
        Maps each node ID → total produced energy in that window.
    """
    prod_per_node = {}
    # 1) Loop over every node ID
    for node_idx in data.node.index.tolist():
        # 2) Get all generators at that node
        node = data.node.loc[node_idx, 'id']
        print("Fetching production for node:", node)
        gens = data.getGeneratorsAtNode(node_idx)
        if not gens:
            # no gens → zero production
            prod_per_node[node] = 0.0
            continue
        # 3) Fetch each generator’s time‑series (all at once)
        series = database.getResultGeneratorPower(gens, time_Prod)
        prod_per_node[node] = sum(series) if series else 0.0
    return prod_per_node


def collectProductionForAllNodesFromDB(data: GridData, database: Database, time_Prod):
    """
    Returns total production per node over the given time range using a single query.

    Parameters
    ----------
    data : GridData
        Contains node list & generator-to-node mapping.
    database : Database
        Database object with getAllGeneratorPower method.
    time_Prod : list
        [min, max] time window for which to fetch generation.

    Returns
    -------
    prod_per_node : dict[int, float]
        Maps each node ID to total produced energy in that window.
    """
    # Get generator-to-node mapping
    gen_to_node = dict(zip(data.generator.index, data.generator['node']))
    generator_indices = list(data.generator.index)

    # Fetch all generator power in one query
    print("Fetching power data for all generators...")
    power_rows = database.getAllGeneratorPowerTest(time_Prod, generator_indices)

    # Initialize production dictionary for all nodes
    prod_per_node = {node_id: 0.0 for node_id in data.node['id']}

    # Expected timesteps
    expected_timesteps = range(time_Prod[0], time_Prod[-1])
    num_timesteps = len(expected_timesteps)

    # Group power by generator index
    power_dict = {}
    for timestep, gen_index, output in power_rows:
        if gen_index not in power_dict:
            power_dict[gen_index] = [float('nan')] * num_timesteps
        if time_Prod[0] <= timestep < time_Prod[-1]:
            power_dict[gen_index][timestep - time_Prod[0]] = output

    # Allocate power to nodes
    for gen_index in generator_indices:
        node_id = gen_to_node.get(gen_index)
        if node_id is None:
            print(f"Generator {gen_index} not mapped to any node.")
            continue
        power = power_dict.get(gen_index, [float('nan')] * num_timesteps)
        total_power = sum(p for p in power if not pd.isna(p))
        prod_per_node[node_id] = prod_per_node.get(node_id, 0.0) + total_power
        print(f"Allocated {total_power} MW to node {node_id} from generator {gen_index}")

    return prod_per_node




def getDemandPerAreaFromDB(data: GridData, db: Database, area, timeMaxMin):
    """
    Returns demand timeseries for given area, as dictionary fields "fixed", "flex", and "sum"

    Parameters
    ----------
    area : str
        The area.
    timeMaxMin : list (default = None)
        [min, max] - lower and upper time interval

    Returns
    -------
    demand_per_area : dict
        The demand per area.
    """
    timerange = range(timeMaxMin[0], timeMaxMin[-1])

    consumer = data.consumer

    dem = [0] * len(timerange)
    flex_demand = [0] * len(timerange)
    consumers = data.getConsumersPerArea()[area]
    for i in consumers:
        ref_profile = consumer.demand_ref[i]
        # accumulate demand for all consumers in this area:
        dem = [
            dem[t - timerange[0]]
            + consumer.demand_avg[i]
            * (1 - consumer.flex_fraction[i])
            * data.profiles[ref_profile][t - timerange[0]]
            for t in timerange
        ]
        flex_demand_i = db.getResultFlexloadPower(i, timeMaxMin)
        if len(flex_demand_i) > 0:
            flex_demand = [sum(x) for x in zip(flex_demand, flex_demand_i)]
    sum_demand = [sum(x) for x in zip(dem, flex_demand)]
    demand_per_area = {"fixed": dem, "flex": flex_demand, "sum": sum_demand}
    return demand_per_area

def getDemandPerZoneFromDB(data: GridData, db: Database, area, zone, timeMaxMin):
    """
    Returns demand timeseries for given zone, as dictionary fields "fixed", "flex", and "sum"

    Parameters
    ----------
    area : str
        The area.
    timeMaxMin : list (default = None)
        [min, max] - lower and upper time interval

    Returns
    -------
    demand_per_area : dict
        The demand per zone.
    """
    timerange = range(timeMaxMin[0], timeMaxMin[-1])

    consumer = data.consumer

    dem = [0] * len(timerange)
    flex_demand = [0] * len(timerange)
    consumers = data.getConsumersPerArea()[area]
    for i in consumers:
        if zone in data.consumer.node[i]:
            ref_profile = consumer.demand_ref[i]
            # accumulate demand for all consumers in this area:
            dem = [
                dem[t - timerange[0]]
                + consumer.demand_avg[i]
                * (1 - consumer.flex_fraction[i])
                * data.profiles[ref_profile][t - timerange[0]]
                for t in timerange
            ]
            flex_demand_i = db.getResultFlexloadPower(i, timeMaxMin)
            if len(flex_demand_i) > 0:
                flex_demand = [sum(x) for x in zip(flex_demand, flex_demand_i)]
    sum_demand = [sum(x) for x in zip(dem, flex_demand)]
    demand_per_zone = {"fixed": dem, "flex": flex_demand, "sum": sum_demand}
    return demand_per_zone

def getDemandPerNodeFromDB(data: GridData, db: Database, area, node, timeMaxMin):
    """
    Returns demand timeseries for given zone, as dictionary fields "fixed", "flex", and "sum"

    Parameters
    ----------
    area : str
        The area.
    timeMaxMin : list (default = None)
        [min, max] - lower and upper time interval

    Returns
    -------
    demand_per_area : dict
        The demand per zone.
    """
    timerange = range(timeMaxMin[0], timeMaxMin[-1])

    consumer = data.consumer

    dem = [0] * len(timerange)
    flex_demand = [0] * len(timerange)
    consumers = data.getConsumersPerArea()[area]
    for i in consumers:
        if node == data.consumer.node[i]:
            ref_profile = consumer.demand_ref[i]
            # accumulate demand for all consumers in this area:
            dem = [
                dem[t - timerange[0]]
                + consumer.demand_avg[i]
                * (1 - consumer.flex_fraction[i])
                * data.profiles[ref_profile][t - timerange[0]]
                for t in timerange
            ]
            flex_demand_i = db.getResultFlexloadPower(i, timeMaxMin)
            if len(flex_demand_i) > 0:
                flex_demand = [sum(x) for x in zip(flex_demand, flex_demand_i)]
    sum_demand = [sum(x) for x in zip(dem, flex_demand)]
    demand_per_node = {"fixed": dem, "flex": flex_demand, "sum": sum_demand}
    return demand_per_node



def collectDemandForAllNodesFromDB(data: GridData, db: Database, timeMaxMin):
    """
    Returns demand timeseries for given zone, as dictionary fields "fixed", "flex", and "sum"

    Parameters
    ----------
    timeMaxMin : list (default = None)
        [min, max] - lower and upper time interval

    Returns
    -------
    total_per_node : dict
        The total demand per node, linked to node id
    """
    timerange = range(timeMaxMin[0], timeMaxMin[-1])
    consumer = data.consumer
    demands_per_node = {}
    node = data.consumer.node

    # assume getConsumersPerArea() returns { area_code: [id, id, ...], ... }
    for area, id_list in data.getConsumersPerArea().items():
        for i in id_list:
            # --- 1) fixed demand profile for node i ---
            ref_profile = consumer.demand_ref[i]
            fixed_i = [
                consumer.demand_avg[i]
                * (1 - consumer.flex_fraction[i])
                * data.profiles[ref_profile][t - timeMaxMin[0]]
                for t in timerange
            ]

            # --- 2) flex demand profile for node i (or zeros if none) ---
            flex_i = db.getResultFlexloadPower(i, timeMaxMin)
            if not flex_i:
                flex_i = [0] * len(timerange)

            # --- 3) sum them up ---
            sum_i = [f + x for f, x in zip(fixed_i, flex_i)]

            # --- 4) store in our master dict ---
            demands_per_node[i] = {
                "node": node[i],
                "fixed": fixed_i,
                "flex": flex_i,
                "sum": sum_i
            }

    # 2) now compute the total (scalar) of the "sum" series for each node
    total_per_node = {
        info["node"]: sum(info["sum"])
        for info in demands_per_node.values()
    }

    return total_per_node

def collectDemandForAllNodesAllTimeStepsFromDB(data: GridData, db: Database, timeMaxMin):
    """
    Returns demand timeseries for given zone, as dictionary fields "fixed", "flex", and "sum"

    Parameters
    ----------
    timeMaxMin : list (default = None)
        [min, max] - lower and upper time interval

    Returns
    -------
    total_per_node : dict
        The total demand per node, linked to node id
    """
    timerange = range(timeMaxMin[0], timeMaxMin[-1])
    consumer = data.consumer
    demands_per_node = {}
    node = data.consumer.node

    # assume getConsumersPerArea() returns { area_code: [id, id, ...], ... }
    for area, id_list in data.getConsumersPerArea().items():
        for i in id_list:
            # --- 1) fixed demand profile for node i ---
            ref_profile = consumer.demand_ref[i]
            fixed_i = [
                consumer.demand_avg[i]
                * (1 - consumer.flex_fraction[i])
                * data.profiles[ref_profile][t - timeMaxMin[0]]
                for t in timerange
            ]

            # --- 2) flex demand profile for node i (or zeros if none) ---
            flex_i = db.getResultFlexloadPower(i, timeMaxMin)
            if not flex_i:
                flex_i = [0] * len(timerange)

            # --- 3) sum them up ---
            sum_i = [f + x for f, x in zip(fixed_i, flex_i)]

            # --- 4) store in our master dict ---
            demands_per_node[i] = {
                "node": node[i],
                "fixed": fixed_i,
                "flex": flex_i,
                "sum": sum_i
            }

    demand = {
        info["node"]: info["sum"]
        for info in demands_per_node.values()
    }

    return demand



def get_production_by_type_FromDB(data: GridData, db: Database, area_OP, time_max_min, DATE_START):
    # Todo: Denne bruker gammel løsning for å håndtere tid. Må fikses.
    """
    Get production by type from the database.

    Parameters
    ----------
    data : dict
        The data dictionary.
    db : Database
        The database object.
    area_OP : str
        The area.
    time_max_min : list
        The time interval.
    DATE_START : str
        The start date.

    Returns
    -------
    df_gen_resampled : pd.DataFrame
        The resampled generation DataFrame.
    df_prices_resampled : pd.DataFrame
        The resampled price DataFrame.
    total_production : float
        The total production.
    """
    time_period = time_max_min[-1] - time_max_min[0]
    # Get Generation by type
    # List of generation types to extract
    generation_types = ['hydro', 'ror', 'nuclear', 'wind_on', 'wind_off', 'solar', 'fossil_gas', 'fossil_other', 'biomass']

    # Dictionary to store production data
    generation_data = {}

    # Iterate through generation types and fetch data
    for gen_type in generation_types:
        try:
            gen_idx = data.getGeneratorsPerAreaAndType()[area_OP].get(gen_type, None)
            if gen_idx:
                production = pd.DataFrame(db.getResultGeneratorPower(gen_idx, time_max_min)).sum(axis=1)
                if production.sum() > 0:  # Ensure we only include nonzero production
                    generation_data[f"{gen_type.capitalize()}"] = production
        except Exception as e:
            print(f"Warning: Could not fetch data for {gen_type} in {area_OP}. Error: {e}")

    # Get Load Demand
    load_demand = getDemandPerAreaFromDB(data, db, area=area_OP, timeMaxMin=time_max_min)

    # Get Avg Price for Area
    nodes_in_area = data.node[data.node['area'] == area_OP].index.tolist()
    node_prices_3 = pd.DataFrame({
        node: getNodalPricesFromDB(db, node=node, timeMaxMin=time_max_min) for node in nodes_in_area
    })
    node_prices_3.index = pd.date_range(DATE_START, periods=time_period, freq='h')
    avg_area_prices = node_prices_3.sum(axis=1) / len(nodes_in_area)

    # Create DataFrame with dynamically collected generation data
    df_gen = pd.DataFrame(generation_data)
    df_gen['Load'] = load_demand['sum']
    df_gen.index = pd.date_range(DATE_START, periods=time_period, freq='h')

    # Define resampling rules dynamically
    resampling_rules = {col: 'sum' for col in df_gen.columns}

    # Resample the data based on the defined rules
    df_gen_resampled = df_gen.resample('7D').agg(resampling_rules)

    # Create price DataFrame
    df_prices = pd.DataFrame({'Price': avg_area_prices})
    df_prices.index = pd.date_range(DATE_START, periods=time_period, freq='h')
    df_prices_resampled = df_prices.resample('1D').agg({'Price': 'mean'})

    total_production = df_gen_resampled.drop('Load', axis=1).sum(axis=1).sum()

    return df_gen_resampled, df_prices_resampled, total_production


# NEW Einar

def get_production_by_type_ideal_timestep(data: GridData, db: Database, area_OP, n_timesteps: int):
    """
    Hent produksjon og prisdata basert på ideelle tidssteg (ikke reelle datoer).

    Parametre
    ---------
    data : GridData
        Datastruktur fra modellen.
    db : Database
        Databasen.
    area_OP : str
        Områdekode.
    n_timesteps : int
        Antall tidssteg (f.eks. 8766.4 * 30 = 262992 for 30 "ideelle" år).

    Returnerer
    ---------
    df_gen : pd.DataFrame
        Produksjon per type + last, indeksert på timesteps.
    df_prices : pd.DataFrame
        Gjennomsnittspris, indeksert på timesteps.
    total_production : float
        Sum av all produksjon.
    """
    # Fast ideal tidsakse (0 til n_timesteps-1)
    time_index = pd.timedelta_range(start="0h", periods=n_timesteps, freq="h")

    generation_types = ['hydro', 'ror', 'nuclear', 'wind_on', 'wind_off', 'solar', 'fossil_gas', 'fossil_other', 'biomass']
    generation_data = {}

    for gen_type in generation_types:
        try:
            gen_idx = data.getGeneratorsPerAreaAndType()[area_OP].get(gen_type, None)
            if gen_idx:
                production = pd.DataFrame(db.getResultGeneratorPower(gen_idx, list(range(n_timesteps)))).sum(axis=1)
                if production.sum() > 0:
                    generation_data[f"{gen_type.capitalize()}"] = production
        except Exception as e:
            print(f"Warning: Could not fetch data for {gen_type} in {area_OP}. Error: {e}")

    # Last
    load_demand = getDemandPerAreaFromDB(data, db, area=area_OP, timeMaxMin=list(range(n_timesteps)))

    # Pris
    nodes_in_area = data.node[data.node['area'] == area_OP].index.tolist()
    node_prices = pd.DataFrame({
        node: getNodalPricesFromDB(db, node=node, timeMaxMin=list(range(n_timesteps))) for node in nodes_in_area
    })
    avg_area_prices = node_prices.sum(axis=1) / len(nodes_in_area)

    # DataFrames
    df_gen = pd.DataFrame(generation_data)
    df_gen['Load'] = load_demand['sum']

    df_prices = pd.DataFrame({'Price': avg_area_prices})

    # === FIX: Trim alle datakilder til samme lengde ===
    min_len = min(len(df_gen), len(df_prices), len(time_index))
    df_gen = df_gen.iloc[:min_len]
    df_prices = df_prices.iloc[:min_len]
    time_index = time_index[:min_len]

    # Sett felles indeks
    df_gen.index = time_index
    df_prices.index = time_index

    total_production = df_gen.sum().sum()

    # === AGGREGER PER IDEALT ÅR ===
    timesteps_per_year = 8766.36667
    n_years = min_len // timesteps_per_year
    base_year = 1991
    year_list = [base_year + i // timesteps_per_year for i in range(min_len)]

    df_gen_yearly = df_gen.copy()
    df_gen_yearly["Year"] = year_list
    df_gen_per_year = df_gen_yearly.groupby("Year").sum()

    return df_gen, df_prices, total_production, df_gen_per_year



# def get_capture_price_by_type_per_node(data: GridData, db: Database, nodes: list, time_range: tuple):



# def get_capture_price_by_type_per_node(data: GridData, db: Database, nodes: list, time_range: tuple):



# def get_capture_price_by_type_per_node(data: GridData, db: Database, nodes: list, time_range: tuple):










def get_total_production_by_type_per_node(data: GridData, db: Database, nodes, time_max_min):
    """
    Get total production per generation type for each specified node.

    Parameters
    ----------
    data : GridData
        PowerGAMA GridData object with generator and node mapping.
    db : Database
        Database object providing access to result data.
    nodes : list of str
        List of node names to include in the aggregation.
    time_max_min : list of int
        Time range [start, end) for the query.

    Returns
    -------
    pd.DataFrame
        A DataFrame with one row per node and columns for each generation type,
        representing total production in MWh.
    """
    result = []

    try:
        generators_df = data.generator.copy()
        generators_df['node'] = generators_df['node'].astype(str)

        # Iterate over each node
        for node in nodes:
            node_data = defaultdict(float)
            node_generators = generators_df[generators_df['node'] == node]

            # Iterate over types for this node
            for gen_type in node_generators['type'].unique():
                gen_subset = node_generators[node_generators['type'] == gen_type]
                generator_indices = gen_subset.index.tolist()

                if generator_indices:
                    power_series = db.getResultGeneratorPower(generator_indices, time_max_min)
                    total_production = sum(power_series)
                    node_data[gen_type.capitalize()] = total_production

            # Add node info
            node_data['Node'] = node
            result.append(node_data)

    except Exception as e:
        print(f"Error fetching production data: {e}")

    return pd.DataFrame(result).set_index('Node')



def get_production_by_type_FromDB_ZoneLevel(data: GridData, db: Database, area, time_max_min, DATE_START, week=True):
    time_period = time_max_min[-1] - time_max_min[0]
    print("Analyzing production by type in", area)

    # Step 1: Create zone-to-node mapping
    node_zone_map = {z: set(n for n, zone in zip(data.node.id, data.node.zone) if zone == z)
                     for z in set(data.node.zone) if area in z}

    # Step 2: Get generator indices per area and type
    generators_per_type = data.getGeneratorsPerAreaAndType().get(area, {})

    # Step 3: Extract generator information (only needed columns)
    generator_data = data.generator[['node', 'type']]

    # Step 4: Group generator indices by zone and type in a dictionary
    zone_gen_map = {
        zone: {
            gt: generator_data.loc[(generator_data['node'].isin(nodes)) & (generator_data['type'] == gt)].index.tolist()
            for gt in generators_per_type.keys()}
        for zone, nodes in node_zone_map.items()
    }

    # Step 5: Fetch production data in a single query per (zone, type)
    zone_production = {zone: {gt: [] for gt in generators_per_type.keys()} for zone in node_zone_map.keys()}

    # Iterate over zones and fetch production data
    for zone, gen_types in zone_gen_map.items():
        for gt, gen_idx in gen_types.items():
            if gen_idx:
                try:
                    # Fetch the accumulated production for the entire zone's generators of this type
                    print(f"Fetching data for {gt} in {zone}")
                    zone_production[zone][gt] = db.getResultGeneratorPower(gen_idx, time_max_min)

                except Exception as e:
                    print(f"Warning: Could not fetch data for {gt} in {area}. Error: {e}")

        # Get Load Demand (ensure this data is fetched for each zone)
        try:
            load_demand = getDemandPerZoneFromDB(data, db, area=area, zone=zone, timeMaxMin=time_max_min)
            zone_production[zone]['Load'] = load_demand['sum']  # Store the summed load demand in 'Load'
            print(f"Fetched Load demand for {zone}")
        except Exception as e:
            print(f"Warning: Could not fetch Load demand for {zone}. Error: {e}")
            zone_production[zone]['Load'] = []  # Default empty list if fetching fails

    # Flatten dictionary into a DataFrame
    df_gen = pd.DataFrame([
        {'Zone': zone, 'GenerationType': gen_type, 'Timestamp': t, 'Production': value}
        for zone, gen_dict in zone_production.items()
        for gen_type, values in gen_dict.items()
        for t, value in enumerate(values)  # Ensure full time series
    ])

    df_gen['Timestamp'] = pd.date_range(DATE_START, periods=time_period, freq='h')[df_gen['Timestamp']]
    df_gen.set_index('Timestamp', inplace=True)

    # Pivot table to create a time-series format with GenerationType as columns
    df_gen_pivot = df_gen.pivot_table(index='Timestamp', columns=['Zone', 'GenerationType'], values='Production', aggfunc='sum')

    if week:
        # Define resampling rules (sum over 7-day periods)
        resampling_rules = {col: 'sum' for col in df_gen_pivot.columns}

        # Resample data to 7-day intervals
        df_gen_resampled = df_gen_pivot.resample('7D').agg(resampling_rules)
    else:
        df_gen_resampled = df_gen_pivot

    # total_production = df_gen_resampled.sum().sum()

    return df_gen_resampled# , total_production




def get_production_by_type_FromDB_NodesInZone(data: GridData, db: Database, zone, time_max_min, DATE_START, week=True):
    time_period = time_max_min[-1] - time_max_min[0]
    print("Analyzing production by type for nodes in", zone)
    area = zone[0:2] # Two first letters

    # Step 1: Create zone-to-node mapping
    node_zone_map = {z: set(n for n, zone in zip(data.node.id, data.node.zone) if zone == z)
                     for z in set(data.node.zone) if area in z}

    node_zone_map = node_zone_map[zone]
    # Step 2: Get generator indices per area and type
    generators_per_type = data.getGeneratorsPerAreaAndType().get(area, {})

    # Step 3: Extract generator information (only needed columns)
    generator_data = data.generator[['node', 'type']]


    # Step 4: Group generator indices by zone and type in a dictionary
    node_gen_map = {
        node: {
            gt: generator_data.loc[(generator_data['node']==node) & (generator_data['type'] == gt)].index.tolist()
            for gt in generators_per_type.keys()}
        for node in node_zone_map
    }

    # Step 5: Fetch production data in a single query per (zone, type)
    node_production = {node: {gt: [] for gt in generators_per_type.keys()} for node in node_zone_map}

    # Iterate over zones and fetch production data
    for node, gen_types in node_gen_map.items():
        for gt, gen_idx in gen_types.items():
            if gen_idx:
                try:
                    # Fetch the accumulated production for the entire zone's generators of this type
                    print(f"Fetching data for {gt} in {node}")
                    node_production[node][gt] = db.getResultGeneratorPower(gen_idx, time_max_min)

                except Exception as e:
                    print(f"Warning: Could not fetch data for {gt} in {node}. Error: {e}")


        # Get Load Demand (ensure this data is fetched for each zone)
        try:
            load_demand = getDemandPerNodeFromDB(data, db, area=area, node=node, timeMaxMin=time_max_min)
            node_production[node]['Load'] = load_demand['sum']  # Store the summed load demand in 'Load'
            print(f"Fetched Load demand for {node}")
        except Exception as e:
            print(f"Warning: Could not fetch Load demand for {node}. Error: {e}")
            node_production[node]['Load'] = []  # Default empty list if fetching fails

    # Flatten dictionary into a DataFrame
    df_gen = pd.DataFrame([
        {'Node': node, 'GenerationType': gen_type, 'Timestamp': t, 'Production': value}
        for node, gen_dict in node_production.items()
        for gen_type, values in gen_dict.items()
        for t, value in enumerate(values)  # Ensure full time series
    ])

    df_gen['Timestamp'] = pd.date_range(DATE_START, periods=time_period, freq='h')[df_gen['Timestamp']]
    df_gen.set_index('Timestamp', inplace=True)

    # Pivot table to create a time-series format with GenerationType as columns
    df_gen_pivot = df_gen.pivot_table(index='Timestamp', columns=['Node', 'GenerationType'], values='Production', aggfunc='sum')

    if week:
        # Define resampling rules (sum over 7-day periods)
        resampling_rules = {col: 'sum' for col in df_gen_pivot.columns}

        # Resample data to 7-day intervals
        df_gen_resampled = df_gen_pivot.resample('7D').agg(resampling_rules)
    else:
        df_gen_resampled = df_gen_pivot


    return df_gen_resampled



# Function to collect flow data
def collect_flow_data(db, time_max_min, cross_country_dict, interconnections_capacity, ac=True):
    flow_data = []
    branch_type = "AC" if ac else "DC"
    for branch_index, (node_from, node_to) in cross_country_dict.items():
        print(f"Fetching flow data for {branch_type} branch {branch_index} from {node_from} to {node_to}")
        branch_flows = db.getResultBranchFlow(branch_index, time_max_min, ac=ac)
        max_capacity = interconnections_capacity[branch_index]
        flow_data.append({
            'index': branch_index,
            'type': branch_type,
            'from': node_from,
            'to': node_to,
            'load [MW]': branch_flows,
            'capacity [MW]': max_capacity,
        })
    return flow_data


def getFlowDataOnALLBranches(data: GridData, db: Database, time_max_min):
    """
    Collect flow on ALL connections.
    :param data: GridData
    :param db: Database
    :param time_max_min:
    :return: flow_df
    """
    # print(f'Collecting Flow Data at ALL AC Lines {", ".join([f"{f} → {t}" for f, t in all_AC_branches])}')
    # print(f'Collecting Flow Data at ALL DC Lines {", ".join([f"{f} → {t}" for f, t in all_DC_branches])}')


    # AC_interconnections, DC_interconnections = filter_connections_by_list(data, chosen_connections)
    AC_interconnections = data.branch
    DC_interconnections = data.dcbranch
    AC_interconnections_capacity = AC_interconnections['capacity']
    DC_interconnections_capacity = DC_interconnections['capacity']

    AC_dict = {
        i: (row['node_from'], row['node_to'])
        for i, row in AC_interconnections.iterrows()
    }
    DC_dict = {
        i: (row['node_from'], row['node_to'])
        for i, row in DC_interconnections.iterrows()
    }

    # Collect AC and DC flow data
    flow_data_AC = collect_flow_data(db, time_max_min, AC_dict, AC_interconnections_capacity, ac=True)
    flow_data_DC = collect_flow_data(db, time_max_min, DC_dict, DC_interconnections_capacity, ac=False)

    # Combine data into a single DataFrame
    flow_df = pd.concat([
        pd.DataFrame(flow_data_AC),
        pd.DataFrame(flow_data_DC)
    ], ignore_index=True)

    return flow_df

########    NEW
def collectFlowDataOnALLBranches(data: GridData, db: Database, time_max_min):
    """
    Collect flow on ALL connections using a single database query per branch type.

    Parameters
    ----------
    data : GridData
        Grid data containing branch information.
    db : Database
        Database object.
    time_max_min : list
        [min, max] - lower and upper time interval.

    Returns
    -------
    flow_df : pd.DataFrame
        DataFrame with flow data for all branches.
    """
    # Get AC and DC branch data
    AC_interconnections = data.branch
    DC_interconnections = data.dcbranch
    AC_interconnections_capacity = AC_interconnections['capacity']
    DC_interconnections_capacity = DC_interconnections['capacity']

    # Create dictionaries for branch metadata
    AC_dict = {
        i: (row['node_from'], row['node_to'])
        for i, row in AC_interconnections.iterrows()
    }
    DC_dict = {
        i: (row['node_from'], row['node_to'])
        for i, row in DC_interconnections.iterrows()
    }

    # Get all branch indices
    AC_indices = list(AC_dict.keys())
    DC_indices = list(DC_dict.keys())

    # Query all flow data at once
    print("Fetching flow data for all AC branches...")
    AC_flows = db.getResultBranchFlowForAll(time_max_min, branch_indices=AC_indices, acdc="ac")
    print("Fetching flow data for all DC branches...")
    DC_flows = db.getResultBranchFlowForAll(time_max_min, branch_indices=DC_indices, acdc="dc")

    # Process AC and DC flows
    flow_data_AC = collectFlowData(AC_flows, AC_dict, AC_interconnections_capacity, time_max_min, ac=True)
    flow_data_DC = collectFlowData(DC_flows, DC_dict, DC_interconnections_capacity, time_max_min, ac=False)

    # Combine into a single DataFrame
    flow_df = pd.concat([
        pd.DataFrame(flow_data_AC),
        pd.DataFrame(flow_data_DC)
    ], ignore_index=True)

    return flow_df

def collectFlowData(flow_rows, cross_country_dict, interconnections_capacity, time_max_min, ac=True):
    """
    Process flow data from a single query into a structured format.

    Parameters
    ----------
    flow_rows : list
        List of tuples (timestep, index, flow) from the database.
    cross_country_dict : dict
        Dictionary mapping branch indices to (node_from, node_to).
    interconnections_capacity : pd.Series
        Series of branch capacities.
    time_max_min : list
        [min, max] - time interval for expected timesteps.
    ac : bool
        True for AC branches, False for DC branches.

    Returns
    -------
    flow_data : list
        List of dictionaries with branch flow data.
    """
    branch_type = "AC" if ac else "DC"
    flow_data = []

    # Create a dictionary to group flows by branch index
    flow_dict = {}
    for timestep, branch_index, flow in flow_rows:
        if branch_index not in flow_dict:
            flow_dict[branch_index] = {}
        flow_dict[branch_index][timestep] = flow

    # Expected timesteps
    expected_timesteps = range(time_max_min[0], time_max_min[-1])
    num_timesteps = len(expected_timesteps)

    # Process each branch
    for branch_index, (node_from, node_to) in cross_country_dict.items():
        print(f"Processing {branch_type} branch {branch_index} from {node_from} to {node_to}")
        # Initialize flow list with NaN for all expected timesteps
        branch_flows = [float('nan')] * num_timesteps
        # Fill in available flow data
        if branch_index in flow_dict:
            for timestep, flow in flow_dict[branch_index].items():
                if time_max_min[0] <= timestep < time_max_min[-1]:
                    branch_flows[timestep - time_max_min[0]] = flow
        max_capacity = interconnections_capacity[branch_index]
        flow_data.append({
            'index': branch_index,
            'type': branch_type,
            'from': node_from,
            'to': node_to,
            'load [MW]': branch_flows,
            'capacity [MW]': max_capacity,
        })

    return flow_data

############


def getImportExportFromDB(data: GridData, database: Database, areas=None, timeMaxMin=None, acdc=["ac", "dc"]):
    """Return time series for import and export for a specified area"""
    if areas is None:
        areas = data.getAllAreas()

    df_importexport = pd.DataFrame(index=areas, columns=["import", "export"])
    for area in areas:
        print(area, end=",")
        # find the associated branches (pos = into area)
        # br = self.grid.getInterAreaBranches(area_to=area,acdc='ac')
        # br_p = br['branches_pos']
        # br_n = br['branches_neg']
        # dcbr = self.grid.getInterAreaBranches(area_to=area,acdc='dc')
        # dcbr_p = dcbr['branches_pos']
        # dcbr_n = dcbr['branches_neg']
        flow_in = 0
        flow_out = 0
        for acdc_type in acdc:
            br = data.getInterAreaBranches(area_to=area, acdc=acdc_type)
            br_pos = database.getResultBranches(timeMaxMin, br_indx=br["branches_pos"], acdc=acdc_type)
            br_neg = database.getResultBranches(timeMaxMin, br_indx=br["branches_neg"], acdc=acdc_type)
            if br_pos.shape[0] > 0:
                flow_in += br_pos[br_pos["flow"] > 0]["flow"].sum()
                flow_out -= br_pos[br_pos["flow"] < 0]["flow"].sum()
            if br_neg.shape[0] > 0:
                flow_in -= br_neg[br_neg["flow"] < 0]["flow"].sum()
                flow_out += br_neg[br_neg["flow"] > 0]["flow"].sum()

        # ie =  self.db.getBranchesSumFlow(branches_pos=br_p,branches_neg=br_n,
        #                                 timeMaxMin=timeMaxMin,
        #                                 acdc='ac')
        # DC branches

        #            dcie =  self.db.getBranchesSumFlow(branches_pos=dcbr_p,
        #                                                 branches_neg=dcbr_n,
        #                                                 timeMaxMin=timeMaxMin,
        #                                                 acdc='dc')
        #            import_a = (sum(v for v in ie['pos'] if v>=0)
        #                         +sum(-v for v in ie['neg'] if v<0)
        #                         #+sum(v for v in dcie['pos'] if v>=0)
        #                         #+sum(-v for v in dcie['neg'] if v<0)
        #                         )
        #            export_a = (sum(-v for v in ie['pos'] if v<0)
        #                         +sum(v for v in ie['neg'] if v>=0)
        #                         #+sum(-v for v in dcie['pos'] if v<0)
        #                         #+sum(v for v in dcie['neg'] if v>=0)
        #                         )
        df_importexport.loc[area, "import"] = flow_in
        df_importexport.loc[area, "export"] = flow_out
    print()
    return df_importexport


def getZoneImportExports(data: GridData, flow_data):
    all_nodes = data.node.id
    # TODO: getResultBranchFlowAll denne henter all branch flow med en gang.
    # Initialize dictionaries to store imports and exports between zone pairs
    zone_imports = defaultdict(float)  # Key: (importer_zone, exporter_zone), Value: total import
    zone_exports = defaultdict(float)  # Key: (exporter_zone, importer_zone), Value: total export
    node_names = [node for node in all_nodes]
    node_to_zone = {}
    for node in node_names:
        if '_' in node:
            # Take prefix before first underscore (e.g., 'DK1_3' -> 'DK1', 'SE3_hub_east' -> 'SE3')
            zone = node.split('_')[0]
        else:
            # No underscore (e.g., 'GB', 'DE') -> use full name as zone
            zone = node
        node_to_zone[node] = zone

    # Process each line in flow_data
    for _, row in flow_data.iterrows():
        from_node = row['from']
        to_node = row['to']
        loads = row['load [MW]']  # List of load values

        # Ensure loads is a list or array
        if isinstance(loads, str):
            loads = eval(loads)  # Convert string representation to list if needed
        loads = np.array(loads)

        # Map nodes to their respective zones
        from_zone = node_to_zone[from_node]
        to_zone = node_to_zone[to_node]

        # Skip if the nodes are in the same zone (optional, depending on your needs)
        if from_zone == to_zone:
            continue

        # Define zone pair (order matters for direction)
        zone_pair_forward = (from_zone, to_zone)  # from_zone -> to_zone
        zone_pair_reverse = (to_zone, from_zone)  # to_zone -> from_zone

        # Positive load: flow from 'from' to 'to'
        # - 'from_zone' exports to 'to_zone'
        # - 'to_zone' imports from 'from_zone'
        positive_loads = loads[loads > 0]
        if len(positive_loads) > 0:
            total_positive = sum(positive_loads)
            zone_exports[zone_pair_forward] += total_positive
            zone_imports[zone_pair_reverse] += total_positive

        # Negative load: flow from 'to' to 'from'
        # - 'to_zone' exports to 'from_zone'
        # - 'from_zone' imports from 'to_zone'
        negative_loads = loads[loads < 0]
        if len(negative_loads) > 0:
            total_negative = sum(-negative_loads)  # Absolute value
            zone_exports[zone_pair_reverse] += total_negative
            zone_imports[zone_pair_forward] += total_negative

    # Convert defaultdict to regular dict for cleaner output (optional)
    zone_imports = dict(zone_imports)
    zone_exports = dict(zone_exports)

    return zone_imports, zone_exports



def checkSpilled_vs_ProducedAtGen(database: Database, gen_idx, time_max_min):
    """
    Check spilled vs produced energy at a specific generator. Collect generator idx from grid data/generator
    :param database:
    :param gen_idx:
    :param time_EB:
    :return:
    """
    spilled = database.getResultGeneratorSpilled(gen_idx, time_max_min)
    produced = database.getResultGeneratorPower(gen_idx, time_max_min)
    sum_spilled = sum(spilled)
    sum_produced = sum(produced)
    print("Spilled: ", sum_spilled/1e6, " TWh,  Produced: ", sum_produced/1e6, " TWh")
    return sum_spilled, sum_produced


def getLoadheddingInAreaFromDB(db: Database, area, timeMaxMin=None):
    load_shed = db.getResultLoadheddingInArea(area, timeMaxMin)
    # use asarray to convert None to nan
    load_shed = np.asarray(load_shed, dtype=float)
    return load_shed

def getLoadsheddingPerNodeFromDB(db: Database, timeMaxMin):
    """get loadshedding sum per node"""
    load_shed_per_node = db.getResultLoadheddingSum(timeMaxMin)
    return load_shed_per_node

def getLoadheddingSumsFromDB(data: GridData, db: Database, timeMaxMin=None):
    """get loadshedding sum per area"""
    load_shed_per_node = db.getResultLoadheddingSum(timeMaxMin)
    areas = data.node.area
    all_areas = data.getAllAreas()
    load_shed_sum = dict()
    for a in all_areas:
        load_shed_sum[a] = sum([load_shed_per_node[i] for i in range(len(areas)) if areas[i] == a])

    # load_shed_sum = np.asarray(load_shed_sum,dtype=float)
    return load_shed_sum

def getAverageBranchFlowsFromDB(db: Database, timeMaxMin, branchtype="ac"):
    """
    Average flow on branches over a given time period

    Parameters
    ==========
    timeMaxMin : list (default = None)
        [min, max] - lower and upper time interval
    branchtype : string
        'ac' (default) or 'dc'

    Returns
    =======
    List with values for each branch:
    [flow from 1 to 2, flow from 2 to 1, average absolute flow]
    """
    # branchflow = self.db.getResultBranchFlowAll(timeMaxMin)
    if branchtype == "ac":
        ac = True
    elif branchtype == "dc":
        ac = False
    else:
        raise Exception('Branch type must be "ac" or "dc"')

    avgflow = db.getResultBranchFlowsMean(timeMaxMin, ac)
    # np.mean(branchflow,axis=1)
    return avgflow



def getAverageBranchSensitivityFromDB(db: Database, timeMaxMin, branchtype="ac"):
    """
    Average branch capacity sensitivity over a given time period

    Parameters
    ----------
    timeMaxMin (list) (default = None)
        [min, max] - lower and upper time interval
    branchtype : str
        ac or dc branch type

    Returns
    =======
    1-dim Array of sensitivities (one per branch)
    """
    avgsense = db.getResultBranchSensMean(timeMaxMin, branchtype)
    # use asarray to convert None to nan
    avgsense = np.asarray(avgsense, dtype=float)
    return avgsense



def getAverageUtilisationFromDB(data: GridData, db: Database, timeMaxMin, branchtype="ac"):
    """
    Average branch utilisation over a given time period

    Parameters
    ----------
    timeMaxMin :  (list) (default = None)
        [min, max] - lower and upper time interval
    branchtype : str
        ac or dc branch type

    Returns
    =======
    1-dim Array of branch utilisation (power flow/capacity)
    """
    if branchtype == "ac":
        cap = data.branch.capacity
    elif branchtype == "dc":
        cap = data.dcbranch.capacity
    avgflow = getAverageBranchFlowsFromDB(db, timeMaxMin, branchtype)[2]
    utilisation = [avgflow[i] / cap.iloc[i] for i in range(len(cap))]
    utilisation = np.asarray(utilisation)
    return utilisation


def getGeneratorOutputSumPerAreaFromDB(data: GridData, db: Database, timeMaxMin):
    """
    Description
    Sums up generation per area.

    Parameters
    ----------
    timeMaxMin (list) (default = None)
        [min, max] - lower and upper time interval

    Returns
    =======
    array of dictionary of generation sorted per area
    """
    generation_per_gen = db.getResultGeneratorPowerSum(timeMaxMin)
    areas_per_gen = [data.node.area[data.node.id == n].tolist()[0] for n in data.generator.node]

    allareas = data.getAllAreas()
    generation = dict()
    for a in allareas:
        generation[a] = sum([generation_per_gen[i] for i in range(len(areas_per_gen)) if areas_per_gen[i] == a])

    return generation


def getGeneratorSpilledSumsFromDB(db: Database, timeMaxMin=None):
    """Get sum of spilled inflow for all generators

    Parameters
    ----------
    timeMaxMin (list) (default = None)
        [min, max] - lower and upper time interval
    """
    v = db.getResultGeneratorSpilledSums(timeMaxMin)
    return v

def getGeneratorSpilled(db: Database, generatorindx, timeMaxMin):
    """Get spilled inflow time series for given generator

    Parameters
    ----------
    generatorindx (int)
        index ofgenerator
    timeMaxMin (list) (default = None)
        [min, max] - lower and upper time interval
    """
    v = db.getResultGeneratorSpilled(generatorindx, timeMaxMin)
    return v


def getGeneratorStorageAllFromDB(db: Database, timestep):
    """Get stored energy for all storage generators at given time

    Parameters
    ----------
    timestep : int
        timestep when storage is requested
    """
    v = db.getResultStorageFillingAll(timestep)
    return v

def getGeneratorStorageValuesFromDB(data: GridData, timestep):
    """Get value of stored energy for given time

    Parameters
    ----------
    timestep : int
        when to compute value

    Returns
    -------
    list of int
        Value of stored energy for all storage generators

    The method uses the storage value absolute level (basecost) per
    generator to compute total storage value
    """
    storage_energy = getGeneratorStorageAllFromDB(db, timestep)
    storage_values = data.generator.storage_price
    idx_storage_generators = data.getIdxGeneratorsWithStorage()
    stor_val = [storage_energy[i] * storage_values[v] for i, v in enumerate(idx_storage_generators)]
    return stor_val


def _node2areaFromDB(data: GridData, nodeName):
    """Returns the area of a specified node"""
    # Is handy when you need to access more information about the node,
    # but only the node name is avaiable. (which is the case in the generator file)
    area = data.node.loc[data.node["id"] == nodeName, "area"].iloc[0]
    return area

def _getAreaTypeProductionFromDB(data: GridData, db: Database, area, generatorType, timeMaxMin):
    """
    Returns total production for specified area nd generator type
    """

    print("Looking for generators of type " + str(generatorType) + ", in " + str(area))
    print("Number of generator to run through: " + str(data.generator.numGenerators()))
    totalProduction = 0

    for genNumber in range(0, data.generator.numGenerators()):
        genNode = data.generator.node[genNumber]
        genType = data.generator.type[genNumber]
        genArea = _node2areaFromDB(genNode)
        # print str(genNumber) + ", " + genName + ", " + genNode + ", " + genType + ", " + genArea
        if (genType == generatorType) and (genArea == area):
            # print "\tGenerator is of right type and area. Adding production"
            genProd = sum(db.getResultGeneratorPower(genNumber, timeMaxMin))
            totalProduction += genProd
            # print "\tGenerator production = " + str(genProd)
    return totalProduction


def getAllGeneratorProductionOBSOLETEFromDB(data: GridData, db:Database, timeMaxMin):
    """Returns all production [MWh] for all generators"""

    totGenNumbers = data.generator.numGenerators()
    totalProduction = 0
    for genNumber in range(0, totGenNumbers):
        genProd = sum(db.getResultGeneratorPower(genNumber, timeMaxMin))
        print(str(genProd))
        totalProduction += genProd
        print("Progression: " + str(genNumber + 1) + " of " + str(totGenNumbers))
    return totalProduction





def getEnergyBalanceInAreaFromDB(data: GridData, db:Database, area, spillageGen, resolution="h", fileName=None, timeMaxMin=None, start_date=None):
    """
    Print time series of energy balance in an area, including
    production, spillage, load shedding, storage, pump consumption
    and imports

    Parameters
    ----------
    area : string
        area code
    spillageGen : list
        generator types for which to show spillage (renewables)
    resolution : string
        resolution of output, see pandas:resample
    fileName : string (default=None)
        name of file to export results
    timeMaxMin : list
        time range to consider
    start_date : date string
        date when time series start

    """
    if timeMaxMin is None:
        timeMaxMin = [db.getTimerange()[0], db.getTimerange()[-1]]

    # data resolution in whole seconds (usually, timeDelta=1.0)
    resolutionS = int(data.timeDelta * 3600)

    prod = pd.DataFrame()
    genTypes = data.getAllGeneratorTypes()
    generators = data.getGeneratorsPerAreaAndType()[area]
    pumpIdx = data.getGeneratorsWithPumpByArea()
    if len(pumpIdx) > 0:
        pumpIdx = pumpIdx[area]
    storageGen = data.getIdxGeneratorsWithStorage()
    areaGen = [item for sublist in list(generators.values()) for item in sublist]
    matches = [x for x in areaGen if x in storageGen]
    for gt in genTypes:
        if gt in generators:
            prod[gt] = db.getResultGeneratorPower(generators[gt], timeMaxMin)
            if gt in spillageGen:
                prod[gt + " spilled"] = db.getResultGeneratorSpilled(generators[gt], timeMaxMin)
    prod["load shedding"] = getLoadheddingInAreaFromDB(db, area, timeMaxMin)
    storage = db.getResultStorageFillingMultiple(matches, timeMaxMin, capacity=False)
    if storage:
        prod["storage"] = storage
    if len(pumpIdx) > 0:
        prod["pumped"] = db.getResultPumpPowerMultiple(pumpIdx, timeMaxMin, negative=True)
    prod["net import"] = getNetImportFromDB(data, db, area, timeMaxMin)
    prod.index = pd.date_range(start_date, periods=timeMaxMin[-1] - timeMaxMin[0], freq="{}s".format(resolutionS))
    if resolution != "h":
        prod = prod.resample(resolution, how="sum")
    if fileName:
        prod.to_csv(fileName)
    else:
        return prod




def getAverageInterareaBranchFlowFromDB(db: Database, filename, timeMaxMin):
    """Calculate average flow in each direction and total flow for
    inter-area branches. Requires sqlite version newer than 3.6

    Parameters
    ----------
    filename : string, optional
        if a filename is given then the information is stored to file.
    timeMaxMin : list with two integer values, or None, optional
        time interval for the calculation [start,end]

    Returns
    -------
    List with values for each inter-area branch:
    [flow from 1 to 2, flow from 2 to 1, average absolute flow]
    """

    #        # Version control of database module. Must be 3.7.x or newer
    #        major = int(list(self.db.sqlite_version)[0])
    #        minor = int(list(self.db.sqlite_version)[1])
    #        version = major + minor / 10.0
    #        # print version
    #        if ((major < 4) and (minor < 7)):
    #            print('current SQLite version: {} ({})'
    #                  .format(self.db.sqlite_version,version))
    #            print('getAverageInterareaBranchFlow() requires 3.7.x or newer')
    #            return


    try:
        results = db.getAverageInterareaBranchFlow(timeMaxMin)
    except Exception as err:
        print("Error occured. Maybe because you are using sqlite<3.7")
        raise (err)

    if filename is not None:
        headers = ("branch", "fromArea", "toArea", "average negative flow", "average positive flow", "average flow")
        with open(filename, "wb") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for row in results:
                writer.writerow(row)
    # else:
    #    for x in results:
    #        print(x)

    return results




def getAverageImportExportFromDB(db: Database, area, timeMaxMin):
    """Return average import and export for a specified area"""

    ia = getAverageInterareaBranchFlowFromDB(db, timeMaxMin=timeMaxMin)

    # export: A->B pos flow + A<-B neg flow
    sum_export = sum([b[4] for b in ia if b[1] == area]) - sum([b[3] for b in ia if b[2] == area])
    # import: A->B neg flow + A<-B pos flow
    sum_import = -sum([b[3] for b in ia if b[2] == area]) + sum([b[4] for b in ia if b[2] == area])
    return dict(exp=sum_export, imp=sum_import)






def getNetImportFromDB(data: GridData, db: Database, area, timeMaxMin=None):
    """Return time series for net import for a specified area"""

    # find the associated branches
    br = data.getInterAreaBranches(area_to=area, acdc="ac")
    br_p = br["branches_pos"]
    br_n = br["branches_neg"]
    dcbr = data.getInterAreaBranches(area_to=area, acdc="dc")
    dcbr_p = dcbr["branches_pos"]
    dcbr_n = dcbr["branches_neg"]

    # AC branches
    ie = db.getBranchesSumFlow(branches_pos=br_p, branches_neg=br_n, timeMaxMin=timeMaxMin, acdc="ac")
    # DC branches
    dcie = db.getBranchesSumFlow(branches_pos=dcbr_p, branches_neg=dcbr_n, timeMaxMin=timeMaxMin, acdc="dc")

    if ie["pos"] and ie["neg"]:
        res_ac = [a - b for a, b in zip(ie["pos"], ie["neg"])]
    elif ie["pos"]:
        res_ac = ie["pos"]
    elif ie["neg"]:
        res_ac = [-a for a in ie["neg"]]
    else:
        res_ac = [0] * (timeMaxMin[-1] - timeMaxMin[0])

    if dcie["pos"] and dcie["neg"]:
        res_dc = [a - b for a, b in zip(dcie["pos"], dcie["neg"])]
    elif dcie["pos"]:
        res_dc = dcie["pos"]
    elif dcie["neg"]:
        res_dc = [-a for a in dcie["neg"]]
    else:
        res_dc = [0] * (timeMaxMin[-1] - timeMaxMin[0])

    res = [a + b for a, b in zip(res_ac, res_dc)]
    return res


def getEnergyBalanceInArea(data: GridData, db: Database, area, spillageGen, resolution="h", fileName=None, timeMaxMin=None, start_date=None):
    """
    Print time series of energy balance in an area, including
    production, spillage, load shedding, storage, pump consumption
    and imports

    Parameters
    ----------
    area : string
        area code
    spillageGen : list
        generator types for which to show spillage (renewables)
    resolution : string
        resolution of output, see pandas:resample
    fileName : string (default=None)
        name of file to export results
    timeMaxMin : list
        time range to consider
    start_date : date string
        date when time series start

    """


    # data resolution in whole seconds (usually, timeDelta=1.0)
    resolutionS = int(data.timeDelta * 3600)

    prod = pd.DataFrame()
    genTypes = data.getAllGeneratorTypes()
    generators = data.getGeneratorsPerAreaAndType()[area]
    pumpIdx = data.getGeneratorsWithPumpByArea()
    if len(pumpIdx) > 0:
        pumpIdx = pumpIdx[area]
    storageGen = data.getIdxGeneratorsWithStorage()
    areaGen = [item for sublist in list(generators.values()) for item in sublist]
    matches = [x for x in areaGen if x in storageGen]
    for gt in genTypes:
        if gt in generators:
            prod[gt] = db.getResultGeneratorPower(generators[gt], timeMaxMin)
            if gt in spillageGen:
                prod[gt + " spilled"] = db.getResultGeneratorSpilled(generators[gt], timeMaxMin)
    prod["load shedding"] = getLoadheddingInAreaFromDB(db, area, timeMaxMin)
    storage = db.getResultStorageFillingMultiple(matches, timeMaxMin, capacity=False)
    if storage:
        prod["storage"] = storage
    if len(pumpIdx) > 0:
        prod["pumped"] = db.getResultPumpPowerMultiple(pumpIdx, timeMaxMin, negative=True)
    prod["net import"] = getNetImportFromDB(data, db, area, timeMaxMin)
    prod.index = pd.date_range(start_date, periods=timeMaxMin[-1] - timeMaxMin[0], freq="{}s".format(resolutionS))
    if resolution != "h":
        prod = prod.resample(resolution, how="sum")
    if fileName:
        prod.to_csv(fileName)
    else:
        return prod