"""
This file contains functions that are used to interact with the database.
"""
import powergama.database as db


def getSystemCostFromDB(data, database, time_max_min):
    """
    This function calculates the system cost from the database.

    Parameters
    ----------
    data : dict
        The data dictionary.
    database : object
        The database object.
    time_max_min : int
        The time in minutes.

    Returns
    -------
    systemcost : dict
        The system cost.
    """
    generation_per_gen = database.getResultGeneratorPowerSum(time_max_min)
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