"""
This file contains functions that are used to interact with the database.
"""
import numpy as np
import pandas as pd
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



def getAverageNodalPricesFromDB(db: Database, timeMaxMin):
    """
    Average nodal price over a given time period

    Parameters
    ----------
    db : Database
        The database object.
    timeMaxMin : list (default = None)
            [min, max] - lower and upper time interval

    """
    avg_prices = db.getResultNodalPricesMean(timeMaxMin)
    # use as array to convert None to nan
    avg_prices = np.asarray(avg_prices, dtype=float)
    return avg_prices



def getAreaPricesAverageFromDB(data: GridData, db: Database, areas, timeMaxMin):
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


def getStorageFillingInAreasFromDB(data: GridData, db: Database, areas, generator_type, relative_storage, timeMaxMin):
    """
    Get the storage filling in areas from the database.

    Parameters
    ----------
    data : dict
        The data dictionary.
    db : Database
        The database object.
    areas : list
        List of areas.
    generator_type : str
        The generator type.
    relative_storage : bool
        If True, the relative storage is returned.
    timeMaxMin : list (default = None)
        [min, max] - lower and upper time interval

    Returns
    -------
    filling : dict
        The storage filling.
    """
    storageGen = data.getIdxGeneratorsWithStorage()
    storageTypes = data.generator.type
    nodeNames = data.generator.node
    nodeAreas = data.node.area
    storCapacities = data.generator.storage_cap
    generators = []
    capacity = 0
    for gen in storageGen:
        area = nodeAreas[data.node.id.tolist().index(nodeNames[gen])]
        if area in areas and storageTypes[gen] == generator_type:
            generators.append(gen)
            if relative_storage:
                capacity += storCapacities[gen]
        filling = db.getResultStorageFillingMultiple(generators, timeMaxMin, capacity)
    return filling

