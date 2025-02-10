"""
This file contains functions that are used to interact with the database.
"""
import csv
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




def get_production_by_typeFromDB(data: GridData, db: Database, area_OP, time_max_min, DATE_START):
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

    # Get Generation by type
    genHydro = 'hydro'
    genHydroIdx = data.getGeneratorsPerAreaAndType()[area_OP][genHydro]
    all_hydro_production = pd.DataFrame(db.getResultGeneratorPower(genHydroIdx, time_max_min)).sum(axis=1)

    genWind = 'wind_on'
    genWindIdx = data.getGeneratorsPerAreaAndType()[area_OP][genWind]
    all_wind_production = pd.DataFrame(db.getResultGeneratorPower(genWindIdx, time_max_min)).sum(axis=1)

    genSolar = 'solar'
    genSolarIdx = data.getGeneratorsPerAreaAndType()[area_OP][genSolar]
    all_solar_production = pd.DataFrame(db.getResultGeneratorPower(genSolarIdx, time_max_min)).sum(axis=1)

    genGas = 'fossil_gas'
    genGasIdx = data.getGeneratorsPerAreaAndType()[area_OP][genGas]
    all_gas_production = pd.DataFrame(db.getResultGeneratorPower(genGasIdx, time_max_min)).sum(axis=1)

    # Get Load Demand
    load_demand = getDemandPerAreaFromDB(data, db, area='NO', timeMaxMin=time_max_min)

    # Get Avg Price for Area
    nodes_in_area = data.node[data.node['area'] == area_OP].index.tolist()
    node_prices = pd.DataFrame({node: getNodalPricesFromDB(db, node=node, timeMaxMin=time_max_min) for node in nodes_in_area})
    node_prices.index = pd.date_range(DATE_START, periods=time_max_min[-1], freq='h')
    avg_area_prices = node_prices.sum(axis=1) / len(nodes_in_area)

    # Create DataFrame
    df_gen = pd.DataFrame({
        'Hydro Production': all_hydro_production,
        'Wind Production': all_wind_production,
        'Solar Production': all_solar_production,
        'Gas Production': all_gas_production,
        'Load': load_demand['sum']
    })
    df_gen.index = pd.date_range(DATE_START, periods=time_max_min[-1], freq='h')

    # Resample the data
    df_gen_resampled = df_gen.resample('7D').agg({
        'Hydro Production': 'sum',
        'Wind Production': 'sum',
        'Solar Production': 'sum',
        'Gas Production': 'sum',
        'Load': 'sum'
    })

    df_prices = pd.DataFrame({
        'Price': avg_area_prices
    })
    df_prices.index = pd.date_range(DATE_START, periods=time_max_min[-1], freq='h')
    df_prices_resampled = df_prices.resample('1D').agg({
        'Price': 'mean'
    })

    total_production = sum(all_hydro_production) + sum(all_wind_production) + sum(all_solar_production) + sum(all_gas_production)

    return df_gen_resampled, df_prices_resampled, total_production


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