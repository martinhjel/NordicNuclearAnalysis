# ARCHIVED FUNCTIONS THAT ARE NO LONGER IN USE


def get_time_steps_for_years(selected_years):
    """
    Given a list of specific years, return a dictionary with min and max time step indices for each year.

    Parameters:
    selected_years (list of int): The years to include (between 1991 and 2020).

    list_of_years = get_time_steps_for_years(selected_years=[1993, 2001, 2009, 2018])

    Returns:
    dict: {year: [min_time_step, max_time_step]} for each selected year.
    """
    first_year = 1991
    last_year = 2020
    leap_years = {1992, 1996, 2000, 2004, 2008, 2012, 2016, 2020}

    # Validate input years
    if any(year < first_year or year > last_year for year in selected_years):
        raise ValueError(f"Years must be between {first_year} and {last_year}")

    # Sort the years to process them in order
    selected_years = sorted(selected_years)

    # Initialize tracking of time steps
    min_time_step = 0
    year_time_steps = {}

    # Loop through all years to calculate min/max for each selected year
    for year in range(first_year, last_year + 1):
        year_hours = 8784 if year in leap_years else 8760  # Handle leap years

        if year in selected_years:
            max_time_step = min_time_step + year_hours - 1
            year_time_steps[year] = [min_time_step, max_time_step]

        # Move to next year
        min_time_step += year_hours

    return year_time_steps


def get_time_steps_for_period(start_year, end_year):
    """
    Given a start and end year, return the corresponding min and max time step indices.

    Parameters:
    start_year (int): The starting year of the period (between 1991 and 2020).
    end_year (int): The ending year of the period (between 1991 and 2020).

    # Example usage:
    timeMaxMin_YEAR = get_time_steps_for_period(2000, 2000)

    Returns:
    list: [min_time_step, max_time_step]
    """
    # Define the valid range
    first_year = 1991
    last_year = 2020

    # Leap years that have 8,784 hours instead of 8,760
    leap_years = {1992, 1996, 2000, 2004, 2008, 2012, 2016, 2020}

    # Validate input years
    if start_year < first_year or end_year > last_year or start_year > end_year:
        raise ValueError(f"Years must be between {first_year} and {last_year}, with start_year ≤ end_year")

    # Compute min time step (start of start_year)
    min_time_step = 0
    for y in range(first_year, start_year):
        min_time_step += 8784 if y in leap_years else 8760

    # Compute max time step (end of end_year)
    max_time_step = min_time_step  # Start at the min time step
    for y in range(start_year, end_year + 1):  # Include the last year
        max_time_step += 8784 if y in leap_years else 8760

    return [min_time_step, max_time_step - 1]  # -1 to get the last valid index






def plotInflowInArea(data: GridData, area, date_start, date_end):

    genTypeIdx = data.getGeneratorsPerAreaAndType()[area]['hydro']
    if area in ['SE', 'FI']:
        genTypeIdx.extend(data.getGeneratorsPerAreaAndType()[area].get('ror', []))
    # Inflow
    inflowFactor = data.generator["inflow_fac"][
        genTypeIdx[0]]  # Assuming all generators have the same inflow factor
    inflowProfile = data.generator["inflow_ref"][
        genTypeIdx]  # All generators have different inflow profiles, zone-based
    inflow_df = pd.DataFrame()

    for gen in genTypeIdx:
        prodCap = data.generator['pmax'][gen]  # MW
        inflow_value = data.profiles[inflowProfile[gen]]
        inflow_df[gen] = [i * prodCap * inflowFactor for i in inflow_value]
    inflow = inflow_df.sum(axis=1)
    # Slice the inflow series using index-based selection
    # inflow = inflow.iloc[time_max_min[0]:time_max_min[1]]  # +1 to include the max index
    print(f"Total inflow: {sum(inflow) / 1e6:.2f} TWh")

    # Plot inflow
    # Plot setup
    import matplotlib
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.serif'] = ['cmr10']  # Computer Modern Roman
    matplotlib.rcParams['axes.formatter.use_mathtext'] = True  # Fix cmr10 warning
    matplotlib.rcParams['axes.unicode_minus'] = False  # Fix minus sign rendering
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(inflow.index, inflow.values, label='Inflow', color='blue')
    plt.title('Inflow over Time (NO)')
    plt.xlabel('Year')
    plt.ylabel('Inflow (MWh/h)')
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.show()
    return inflow


def plotInflowInZone(data: GridData, zone, date_start, date_end):

    genTypeIdx = data.getGeneratorsPerZoneAndType()[zone]['hydro']
    if zone in ['SE1','SE2','SE3','SE4','FI']:
        genTypeIdx.extend(data.getGeneratorsPerZoneAndType()[zone].get('ror', []))
    # Inflow
    inflowFactor = data.generator["inflow_fac"][
        genTypeIdx[0]]  # Assuming all generators have the same inflow factor
    inflowProfile = data.generator["inflow_ref"][
        genTypeIdx]  # All generators have different inflow profiles, zone-based
    inflow_df = pd.DataFrame()

    for gen in genTypeIdx:
        prodCap = data.generator['pmax'][gen]  # MW
        inflow_value = data.profiles[inflowProfile[gen]]
        inflow_df[gen] = [i * prodCap * inflowFactor for i in inflow_value]
    inflow = inflow_df.sum(axis=1)
    # Slice the inflow series using index-based selection
    # inflow = inflow.iloc[time_max_min[0]:time_max_min[1]]  # +1 to include the max index
    print(f"Total inflow: {sum(inflow) / 1e6:.2f} TWh")

    # Ensure the index is datetime
    inflow.index = pd.date_range(
        start=f"{date_start['year']}-{date_start['month']}-{date_start['day']} {date_start['hour']}:00",
        end=f"{date_end['year']}-{date_end['month']}-{date_end['day']} {date_end['hour']}:00",
        freq='h')

    # Resample to yearly data (sum or mean, depending on your preference)
    # inflow_yearly = inflow.resample('Y').sum()  # Use 'Y' for year-end frequency, sums hourly inflow for each year

    # Plot inflow
    # Plot setup
    import matplotlib
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.serif'] = ['cmr10']  # Computer Modern Roman
    matplotlib.rcParams['axes.formatter.use_mathtext'] = True  # Fix cmr10 warning
    matplotlib.rcParams['axes.unicode_minus'] = False  # Fix minus sign rendering
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(inflow.index.year, inflow.values, label='Inflow', color='blue')
    plt.title('Inflow over Time (NO)')
    plt.xlabel('hours')
    plt.ylabel('Inflow (MWh/h)')
    plt.legend()
    plt.tight_layout()
    plt.grid()

    # Set x-axis to show integer years
    ax.set_xticks(range(date_start['year'], date_end['year'] + 1, 1))  # Show every year
    ax.set_xticklabels(range(date_start['year'], date_end['year'] + 1, 1), rotation=45)

    plt.show()
    return inflow

START = {"year": 1991, "month": 1, "day": 1, "hour": 0}
END = {"year": 2020, "month": 12, "day": 31, "hour": 23}
time_EB = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
zones = 'NO1'
inflow = plotInflowInZone(data, zones, OUTPUT_PATH_PLOTS, START, END)

# %%
def plotInflowValueBox(data: GridData, zone, OUTPUT_PATH_PLOTS, date_start, date_end):
    # Extract inflow value and time index for the specific zone
    inflow_value = data.profiles[f'inflow_{zone}']  # e.g., 'inflow_NO1'
    time_index = data.profiles['time']

    # inflow_series = pd.Series(inflow_value, index=time_index)

    # Debugging: Verify data
    # print(f"Length of inflow_{zone}: {len(inflow_value)}")
    # print(f"Length of time_index: {len(time_index)}")
    # print(f"Sample inflow values (first 5): {inflow_value[:5]}")
    # print(f"Time index range: {time_index.min()} to {time_index.max()}")

    # Check for non-numeric or NaN values in inflow_value
    inflow_array = np.array(inflow_value, dtype=object)  # Avoid premature type coercion
    non_numeric = [x for x in inflow_array if not isinstance(x, (int, float)) and x is not None]
    print(f"Non-numeric values (first 5, if any): {non_numeric[:5] if non_numeric else 'None'}")

    # Ensure time_index is a DatetimeIndex
    if not isinstance(time_index, pd.DatetimeIndex):
        print("Converting time_index to DatetimeIndex")
        time_index = pd.to_datetime(time_index, errors='coerce')
        if time_index.isna().any():
            print("Error: Some timestamps are invalid")
            return None

    # Create pandas Series
    try:
        inflow_series = pd.Series(inflow_array, index=time_index)
    except Exception as e:
        print(f"Error creating Series: {e}")
        return None

    # Check for NaN in inflow_series
    nan_series_count = inflow_series.isna().sum()
    # Handle NaN values (if any)
    if nan_series_count > 0:
        print("Handling NaN values with linear interpolation")
        inflow_series = inflow_series.interpolate(method='linear')
        inflow_series = inflow_series.fillna(0)  # Fill remaining NaN (e.g., at start/end)
        print(f"Number of NaN values after handling: {inflow_series.isna().sum()}")

    # Verify 30-year average
    mean_inflow = inflow_series.mean()
    print(f"30-year average inflow: {mean_inflow:.4f} (should be ~1)")

    # Calculate percentage deviation from the 30-year average (1)
    percentage_deviation = ((inflow_series - 1) / 1) * 100  # in %
    # Group deviations by year for box plot
    # Create a DataFrame with percentage deviations and year as a column
    deviation_df = pd.DataFrame({
        'deviation': percentage_deviation,
        'year': inflow_series.index.year
    })

    # Prepare data for box plot: group deviations by year
    boxplot_data = [deviation_df[deviation_df['year'] == year]['deviation'].values
                    for year in range(date_start['year'], date_end['year'] + 1)]

    # Plot setup
    import matplotlib
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.serif'] = ['cmr10']  # Computer Modern Roman
    matplotlib.rcParams['axes.formatter.use_mathtext'] = True
    matplotlib.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(15, 6))

    # # Plot yearly inflow
    # ax.plot(time_index, inflow_value, label=f'Inflow ({zone})', color='blue')
    # plt.title(f'Inflow over Time ({zone})')
    # plt.xlabel('Year')
    # plt.ylabel('Normalized Inflow')  # Reflects yearly aggregation
    # plt.legend()
    # plt.tight_layout()
    # plt.grid()
    #
    # # Set x-axis to show every year with 45-degree rotation
    # import matplotlib.dates as mdates
    # ax.xaxis.set_major_locator(mdates.YearLocator(1))  # Every year
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Year format
    # plt.setp(ax.get_xticklabels(), rotation=45)
    # filename = f'InflowProfile_{zone}.pdf'
    # plt.savefig(OUTPUT_PATH_PLOTS / filename)
    # plt.show()
    # Create box plot
    ax.boxplot(boxplot_data, labels=range(date_start['year'], date_end['year'] + 1))
    plt.title(f'Yearly Inflow Deviation from 30-Year Average ({zone})')
    plt.xlabel('Year')
    plt.ylabel('Percentage Deviation (\%)')
    plt.grid(True, axis='y')

    # Rotate x-axis labels by 45 degrees
    plt.setp(ax.get_xticklabels(), rotation=45)

    # Tight layout to prevent label cutoff
    plt.tight_layout()
    plt.show()

    # Calculate and print yearly averages for reference
    yearly_averages = inflow_series.resample('Y').mean()
    yearly_deviations = ((yearly_averages - 1) / 1) * 100
    print("\nYearly Average Inflow and Percentage Deviation from 1:")
    for year, avg, dev in zip(yearly_averages.index.year, yearly_averages, yearly_deviations):
        print(f"{year}: Avg = {avg:.4f}, Deviation = {dev:.2f}%")




def plotInflowValuePlot(data: GridData, zones, OUTPUT_PATH_PLOTS, start, end):
    # Plot setup
    import matplotlib
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.serif'] = ['cmr10']  # Computer Modern Roman
    matplotlib.rcParams['axes.formatter.use_mathtext'] = True
    matplotlib.rcParams['axes.unicode_minus'] = False

    start_date = pd.Timestamp(f"{start['year']}-{start['month']}-{start['day']} {start['hour']}:00",
                              tz='UTC')
    end_date = pd.Timestamp(f"{end['year']}-{end['month']}-{end['day']} {end['hour']}:00", tz='UTC')

    # Create a figure with 5 subplots (one for each zone), stacked vertically
    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(12.405, 17.535), sharex=True)


    # Plot inflow for each zone in its respective subplot
    for i, zone in enumerate(zones):
        # Extract inflow value and time index for the specific zone
        inflow_value = data.profiles[f'inflow_{zone}']  # e.g., 'inflow_NO1'
        time_index = data.profiles['time']

        # Plot in the corresponding subplot
        ax = axes[i]
        ax.plot(time_index, inflow_value, label=f'Inflow ({zone})', color='blue', linewidth=1)
        ax.set_title(f'Inflow over Time ({zone})', fontsize=10)
        ax.set_ylabel('Normalized Inflow', fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True)

        # Set x-axis for the bottom subplot (shared across all subplots)
        ax.set_xlabel('Year', fontsize=10)
        import matplotlib.dates as mdates
        ax.xaxis.set_major_locator(mdates.YearLocator(1))  # Every year
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Year format
        plt.setp(ax.get_xticklabels(), rotation=45)

        # Set x-axis limits to exactly the start and end dates
        ax.set_xlim(start_date, end_date)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the plot (commented out as in your original code)
    filename = 'InflowProfiles_AllZones.pdf'
    plt.savefig(OUTPUT_PATH_PLOTS / filename, dpi=600)
    plt.show()

# === INITIALIZATIONS ===
START = {"year": 1991, "month": 1, "day": 1, "hour": 0}
END = {"year": 2020, "month": 12, "day": 31, "hour": 23}
time_EB = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
zones = ['NO1', 'NO2', 'NO3', 'NO4', 'NO5']
# inflow = plotInflowValueBox(data, zone, OUTPUT_PATH_PLOTS, START, END)
inflow = plotInflowValuePlot(data, zones, OUTPUT_PATH_PLOTS, START, END)









import pandas as pd
from datetime import datetime


def get_zone_production_summary_full_period(data, database, time_Prod, START, END, OUTPUT_PATH):
    '''
    Retrieves production data for the selected nodes over the specified time period,
    returns production by zone and by type for each timestep, converts the results to TWh,
    and merges selected production types into broader categories.

    Parameters:
        SELECTED_NODES (list or str): List of node IDs to include or "ALL" to select all nodes.
        START (dict): Dictionary defining the start time with keys "year", "month", "day", "hour".
        END (dict): Dictionary defining the end time with keys "year", "month", "day", "hour".
        TIMEZONE (str): Timezone name.
        SIM_YEAR_START (datetime): Start of simulation year.
        SIM_YEAR_END (datetime): End of simulation year.
        data (object): Data object containing node information.
        database (object): Database connection or access object for production data.

    Returns:
        zone_summed_df (pd.DataFrame): Production per original production type for each zone, in TWh, with time index.
        zone_summed_merged_df (pd.DataFrame): Production per merged production type for each zone, with total, in TWh, with time index.
    '''
    # Get list of nodes
    Nodes = data.node["id"].dropna().unique().tolist()

    # Get production data
    production_per_node, gen_idx, gen_type = GetProductionAtSpecificNodes(Nodes, data, database, time_Prod[0], time_Prod[1])

    # Create time index
    start_time = datetime(START['year'], START['month'], START['day'], START['hour'], 0, tzinfo=TIMEZONE)
    end_time = datetime(END['year'], END['month'], END['day'], END['hour'], 0, tzinfo=TIMEZONE)
    time_index = pd.date_range(start=start_time, end=end_time, freq='h')
    num_timesteps = len(time_index)

    # Initialize dictionary to store time-series data by zone and production type
    zone_production = {}

    # Process production data
    for node, prodtypes in production_per_node.items():
        zone = node.split("_")[0]  # Extract zone from node ID (e.g., 'SE1' from 'SE1_hydro_1')
        if zone not in zone_production:
            zone_production[zone] = {}

        for prodtype, values_list in prodtypes.items():
            # Handle empty or null values
            if not values_list or not values_list[0]:
                values = [0] * num_timesteps
            else:
                values = values_list[0]  # Assume values_list[0] contains the time-series data
                if len(values) != num_timesteps:
                    raise ValueError(
                        f"Production data for node {node}, type {prodtype} has incorrect length: {len(values)} vs {num_timesteps}")

            # Store time-series data
            if prodtype not in zone_production[zone]:
                zone_production[zone][prodtype] = values
            else:
                # Sum production for the same production type in the same zone
                zone_production[zone][prodtype] = [sum(x) for x in zip(zone_production[zone][prodtype], values)]

    # Convert to DataFrame with multi-level columns (zone, prodtype)
    columns = pd.MultiIndex.from_tuples(
        [(zone, prodtype) for zone in zone_production for prodtype in zone_production[zone]],
        names=['Zone', 'Production Type']
    )
    zone_summed_df = pd.DataFrame(
        data=[[zone_production[zone][prodtype][t] for zone in zone_production for prodtype in zone_production[zone]]
              for t in range(num_timesteps)],
        index=time_index,
        columns=columns
    )

    # Convert from MWh to TWh
    # zone_summed_df = zone_summed_df / 1e6

    # Merge production types
    merge_mapping = {
        "Hydro": ["hydro", "ror"],
        "Nuclear": ["nuclear"],
        "Solar": ["solar"],
        "Thermal": ["fossil_gas", "fossil_other", "biomass"],
        "Wind Onshore": ["wind_on"],
        "Wind Offshore": ["wind_off"]
    }

    # Initialize merged DataFrame
    merged_data = {}
    for zone in zone_production:
        for new_type, old_types in merge_mapping.items():
            # Sum the relevant production types for this zone
            valid_types = [t for t in old_types if t in zone_production[zone]]
            if valid_types:
                merged_data[(zone, new_type)] = zone_summed_df[zone][valid_types].sum(axis=1, skipna=True)

                # Only include non-zero columns
                if merged_data[(zone, new_type)].eq(0).all():
                    del merged_data[(zone, new_type)]

        # Add total production for the zone
        merged_data[(zone, "Production Total")] = zone_summed_df[zone].sum(axis=1, skipna=True)

    # Create merged columns only for existing data
    merged_columns = pd.MultiIndex.from_tuples(
        [col for col in merged_data.keys()],
        names=['Zone', 'Production Type']
    )
    zone_summed_merged_df = pd.DataFrame(
        data={col: merged_data[col] for col in merged_columns},
        index=time_index
    )


    zone_summed_df.to_csv(OUTPUT_PATH / f'zone_summed_df_{start_time.year}_{end_time.year}.csv')
    zone_summed_merged_df.to_csv(OUTPUT_PATH / f'zone_summed_merged_df_{start_time.year}_{end_time.year}.csv')

    return zone_summed_df, zone_summed_merged_df

START = {"year": 2016, "month": 7, "day": 21, "hour": 0}
END = {"year": 2016, "month": 7, "day": 25, "hour": 23}
time_Prod = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
# ======================================================================================================================
# TODO: Valg av merged eller ikke, lage for area, samle alt i excel fil med flere ark

zone_summed_df, zone_summed_merged_df = get_zone_production_summary_full_period(data, database, time_Prod, START, END, OUTPUT_PATH / 'data_files')

# %%


def get_node_production_summary_full_period(data, database, time_Prod, START, END, OUTPUT_PATH):
    '''
    Retrieves production data for the selected nodes over the specified time period,
    returns production by node and by type for each timestep, and merges selected
    production types into broader categories. Discards columns with all zero values.

    Parameters:
        data (object): Data object containing node information.
        database (object): Database connection or access object for production data.
        time_Prod (tuple): Time range for production data.
        START (dict): Dictionary defining the start time with keys "year", "month", "day", "hour".
        END (dict): Dictionary defining the end time with keys "year", "month", "day", "hour".
        OUTPUT_PATH (Path): Path to save output CSV files.

    Returns:
        node_summed_df (pd.DataFrame): Production per original production type for each node, in MWh, with time index.
        node_summed_merged_df (pd.DataFrame): Production per merged production type for each node, with total, in MWh, with time index.
    '''
    from datetime import datetime
    import pandas as pd

    # Get list of nodes
    Nodes = data.node["id"].dropna().unique().tolist()

    # Get production data
    production_per_node, gen_idx, gen_type = GetProductionAtSpecificNodes(Nodes, data, database, time_Prod[0], time_Prod[1])

    # Create time index
    start_time = datetime(START['year'], START['month'], START['day'], START['hour'], 0, tzinfo=TIMEZONE)
    end_time = datetime(END['year'], END['month'], END['day'], END['hour'], 0, tzinfo=TIMEZONE)
    time_index = pd.date_range(start=start_time, end=end_time, freq='h')
    num_timesteps = len(time_index)

    # Initialize dictionary to store time-series data by node and production type
    node_production = {}

    # Process production data
    for node, prodtypes in production_per_node.items():
        if node not in node_production:
            node_production[node] = {}

        for prodtype, values_list in prodtypes.items():
            # Handle empty or null values
            if not values_list or not values_list[0]:
                values = [0] * num_timesteps
            else:
                values = values_list[0]  # Assume values_list[0] contains the time-series data
                if len(values) != num_timesteps:
                    raise ValueError(
                        f"Production data for node {node}, type {prodtype} has incorrect length: {len(values)} vs {num_timesteps}")

            # Store time-series data
            if prodtype not in node_production[node]:
                node_production[node][prodtype] = values
            else:
                # Sum production for the same production type in the same node
                node_production[node][prodtype] = [sum(x) for x in zip(node_production[node][prodtype], values)]

    # Convert to DataFrame with multi-level columns (node, prodtype)
    columns = pd.MultiIndex.from_tuples(
        [(node, prodtype) for node in node_production for prodtype in node_production[node]],
        names=['Node', 'Production Type']
    )
    node_summed_df = pd.DataFrame(
        data=[[node_production[node][prodtype][t] for node in node_production for prodtype in node_production[node]]
              for t in range(num_timesteps)],
        index=time_index,
        columns=columns
    )

    # Discard columns with all zero values
    # node_summed_df = node_summed_df.loc[:, (node_summed_df != 0).any(axis=0)]

    # Merge production types
    merge_mapping = {
        "Hydro": ["hydro", "ror"],
        "Nuclear": ["nuclear"],
        "Solar": ["solar"],
        "Thermal": ["fossil_gas", "fossil_other", "biomass"],
        "Wind Onshore": ["wind_on"],
        "Wind Offshore": ["wind_off"]
    }

    # Initialize merged DataFrame
    merged_data = {}
    for node in node_production:
        for new_type, old_types in merge_mapping.items():
            # Only include production types that exist in node_summed_df[node]
            valid_types = [t for t in old_types if (node, t) in node_summed_df.columns]
            if valid_types:
                merged_data[(node, new_type)] = node_summed_df[node][valid_types].sum(axis=1, skipna=True)
                # Only include non-zero columns
                if merged_data[(node, new_type)].eq(0).all():
                    del merged_data[(node, new_type)]

        # Add total production for the node
        merged_data[(node, "Production total")] = node_summed_df[node].sum(axis=1, skipna=True)

    # Create merged columns only for existing data
    merged_columns = pd.MultiIndex.from_tuples(
        [col for col in merged_data.keys()],
        names=['Node', 'Production Type']
    )
    node_summed_merged_df = pd.DataFrame(
        data={col: merged_data[col] for col in merged_columns},
        index=time_index
    )

    # Discard columns with all zero values in merged DataFrame
    node_summed_merged_df = node_summed_merged_df.loc[:, (node_summed_merged_df != 0).any(axis=0)]

    # Save DataFrames to CSV
    node_summed_df.to_csv(OUTPUT_PATH / f'node_summed_df_{start_time.year}_{end_time.year}.csv')
    node_summed_merged_df.to_csv(OUTPUT_PATH / f'node_summed_merged_df_{start_time.year}_{end_time.year}.csv')

    return node_summed_df, node_summed_merged_df


# Example usage
START = {"year": 1992, "month": 10, "day": 1, "hour": 0}
END = {"year": 1992, "month": 12, "day": 31, "hour": 23}
time_Prod = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
node_summed_df, node_summed_merged_df = get_node_production_summary_full_period(
    data, database, time_Prod, START, END, OUTPUT_PATH / 'data_files'
)




# %% Load, generation by type in AREA

# Siden det er ukes aggregert, bør man trekke fra 23 timer for vanlig år, og 1d23t for skuddår
# for et godt plot.
# === INITIALIZATIONS ===
START = {"year": 1994, "month": 1, "day": 1, "hour": 0}
END = {"year": 1994, "month": 12, "day": 31, "hour": 0}

# === PLOT CONFIGURATIONS ===
plot_config = {
    'area': 'GB',
    'title': 'Production, Consumption and Price in GB',
    "fig_size": (15, 10),
    "plot_full_timeline": True,
    "duration_curve": False,
    "box_in_frame": False,
    "save_fig": True,                      # True: Save plot as pdf
    "interval": 1                           # Number of months on x-axis. 1 = Step is one month, 12 = Step is 12 months
}


# === COMPUTE TIMERANGE AND PLOT FLOW ===
time_LGT = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
df_gen_re, df_prices, tot_prod = calcPlot_LG_FromDB(data, database, time_LGT, OUTPUT_PATH_PLOTS, DATE_START, plot_config)
tot_prod = df_gen_re.drop('Load', axis=1).sum(axis=1).sum()
print(f"Total production in {plot_config['area']}: {tot_prod:.2f} MWh")



# %%

# === INITIALIZATIONS ===
START = {"year": 1993, "month": 1, "day": 1, "hour": 0}
END = {"year": 1993, "month": 12, "day": 31, "hour": 23}
area = None
zone = 'NO2'

# Juster area for å se på sonene, og zone for å se på nodene i sonen
# === COMPUTE TIMERANGE AND PLOT FLOW ===
time_Prod = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
correct_date_start_Prod = DATE_START + pd.Timedelta(hours=time_Prod[0])
if area is not None:
    zones_in_area_prod = getProductionZonesInArea(data, database, area, time_Prod, OUTPUT_PATH, correct_date_start_Prod, week=True)
    energyBalanceZones = zones_in_area_prod.sum(axis=0)

if zone is not None:
    nodes_in_zone_prod = getProductionNodesInZone(data, database, zone, time_Prod, OUTPUT_PATH, correct_date_start_Prod, week=True)
    energyBalanceNodes = nodes_in_zone_prod.sum(axis=0)



#%% Excel ###
"""
Production, consumption, and price data for specific nodes within a given time period.

Main Features:
- Handles time using Python's built-in datetime objects.
- Retrieves simulated production, consumption, and price data from a given SQL file for selected nodes within a specified timeframe.
- Organizes data and exports it to an Excel file for further analysis.
"""

# === INITIALIZATIONS ===
START = {"year": 1992, "month": 1, "day": 1, "hour": 0}
END = {"year": 1992, "month": 12, "day": 31, "hour": 23}
Nodes = ["DK1_2", "DK1_3", "DK2_2", "DK2_1", "DK2_hub", "SE4_2", "SE3_7", "SE3_9", "DE"]
SELECTED_BRANCHES  = [['DK2_hub','DK2_2'], ['SE4_2', 'DE'], ['DK2_2', 'SE4_2'], ['SE3_7', 'DK1_1'], ['DK2_2', 'DE'], ['DK2_hub','DE'], ['DE','DK1_3']]
# ======================================================================================================================

start_hour, end_hour = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
production_per_node, gen_idx, gen_type = GetProductionAtSpecificNodes(Nodes, data, database, start_hour, end_hour)


consumption_per_node = GetConsumptionAtSpecificNodes(Nodes, data, database, start_hour, end_hour)
nodal_prices_per_node = GetPriceAtSpecificNodes(Nodes, data, database, start_hour, end_hour)
reservoir_filling_per_node, storage_cap = GetReservoirFillingAtSpecificNodes(Nodes, data, database, start_hour, end_hour)
flow_data = getFlowDataOnBranches(data, database, [start_hour, end_hour], SELECTED_BRANCHES)
excel_filename = ExportToExcel(Nodes, production_per_node, consumption_per_node, nodal_prices_per_node, reservoir_filling_per_node, storage_cap, flow_data, START, END, SCENARIO, VERSION, OUTPUT_PATH)


# %% === PDF HANDLING ===

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib

# Enable LaTeX rendering for Computer Modern font
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['cmr10']  # Computer Modern Roman
matplotlib.rcParams['axes.formatter.use_mathtext'] = True  # Fix cmr10 warning
matplotlib.rcParams['axes.unicode_minus'] = False  # Fix minus sign rendering

# Placeholder: Define zones (excluding 'DE') based on previous node-to-zone mapping
node_names = [node for node in all_nodes]
node_to_zone = {}
for node in node_names:
    if '_' in node:
        zone = node.split('_')[0]
    else:
        zone = node
    node_to_zone[node] = zone
all_zones = sorted([zone for zone in set(node_to_zone.values())])

# Placeholder: Generate sample data for each zone (replace with actual data)
time_period = pd.date_range(start="2025-01-01", end="2025-01-02", freq="h")  # Use 'h' for hourly
n_timesteps = len(time_period)

# Sample data structure
zone_data = {}
for zone in all_zones:
    zone_data[zone] = {
        'consumption': pd.DataFrame({
            'time': time_period,
            'demand': np.random.uniform(100, 1000, n_timesteps)  # MW
        }),
        'generation': pd.DataFrame({
            'time': time_period,
            'wind': np.random.uniform(0, 500, n_timesteps),      # MW
            'hydro': np.random.uniform(0, 600, n_timesteps),     # MW
            'thermal': np.random.uniform(0, 400, n_timesteps)    # MW
        }),
        'prices': pd.DataFrame({
            'time': time_period,
            'price': np.random.uniform(20, 100, n_timesteps)     # €/MWh
        }),
        'storage': pd.DataFrame({
            'time': time_period,
            'level': np.random.uniform(0, 1000, n_timesteps)     # MWh
        }) if np.random.rand() > 0.3 else None  # Simulate some zones lacking storage
    }

# PDF settings
pdf_filename = "zone_energy_report.pdf"
page_width, page_height = 8.27, 11.69  # A4 in inches (595 x 842 points at 72 DPI)
margin = 0.25  # Reduced margins (in inches)
plot_width = page_width - 2 * margin  # ~7.77 inches
plot_height = 3.0  # Plot height (in inches, ~216 points)
spacing = 0.2  # Spacing between plots (in inches)
max_page_height = page_height - 2 * margin  # Usable page height (~11.19 inches)

# Initialize PDF
with PdfPages(pdf_filename) as pdf:
    for zone in all_zones:
        current_height = 0  # Track total height used on current page
        figs = []  # Store figures for this zone

        # 1. Consumption Plot
        fig, ax = plt.subplots(figsize=(plot_width, plot_height))
        data = zone_data[zone]['consumption']
        ax.plot(data['time'], data['demand'], color='red', label='Demand')
        ax.set_title(f"Consumption in {zone}", fontsize=12)
        ax.set_xlabel("Time", fontsize=10)
        ax.set_ylabel("Demand (MW)", fontsize=10)
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45, fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        figs.append(fig)

        # 2. Generation by Type Plot
        fig, ax = plt.subplots(figsize=(plot_width, plot_height))
        gen_data = zone_data[zone]['generation']
        for column in gen_data.columns[1:]:  # Skip 'time'
            ax.plot(gen_data['time'], gen_data[column], label=column.capitalize())
        ax.set_title(f"Generation by Type in {zone}", fontsize=12)
        ax.set_xlabel("Time", fontsize=10)
        ax.set_ylabel("Generation (MW)", fontsize=10)
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45, fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        figs.append(fig)

        # 3. Prices Plot
        fig, ax = plt.subplots(figsize=(plot_width, plot_height))
        price_data = zone_data[zone]['prices']
        ax.plot(price_data['time'], price_data['price'], color='blue', label='Price')
        ax.set_title(f"Prices in {zone}", fontsize=12)
        ax.set_xlabel("Time", fontsize=10)
        ax.set_ylabel("Price (€/MWh)", fontsize=10)
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45, fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        figs.append(fig)

        # 4. Storage Filling Plot (if available)
        if zone_data[zone]['storage'] is not None:
            fig, ax = plt.subplots(figsize=(plot_width, plot_height))
            storage_data = zone_data[zone]['storage']
            ax.plot(storage_data['time'], storage_data['level'], color='purple', label='Storage Level')
            ax.set_title(f"Storage Filling in {zone}", fontsize=12)
            ax.set_xlabel("Time", fontsize=10)
            ax.set_ylabel("Storage (MWh)", fontsize=10)
            ax.legend()
            ax.grid(True)
            plt.xticks(rotation=45, fontsize=8)
            plt.yticks(fontsize=8)
            plt.tight_layout()
            figs.append(fig)

        # Save figures to PDF, grouping by zone
        for i, fig in enumerate(figs):
            # Calculate height needed for this plot (plot_height + spacing, except for last plot)
            plot_total_height = plot_height + (spacing if i < len(figs) - 1 else 0)

            # Check if plot fits on current page
            if current_height + plot_total_height > max_page_height and current_height > 0:
                pdf.savefig(bbox_inches='tight')  # Save current page
                current_height = 0  # Reset for new page

            pdf.savefig(fig, bbox_inches='tight')  # Save figure
            current_height += plot_total_height  # Update height used
            plt.close(fig)  # Close figure to free memory

        # Reset for next zone (no extra pdf.savefig needed)
        current_height = 0

print(f"PDF generated: {pdf_filename}")



# %% PLOT ZONAL PRICES

# === INITIALIZATIONS ===
START = {"year": 1994, "month": 1, "day": 1, "hour": 0}
END = {"year": 1994, "month": 12, "day": 31, "hour": 23}

# === PLOT CONFIGURATIONS ===
plot_config = {
    'zones': ['NO2'],                       # Zones for plotting
    "plot_by_year": True,                   # (True)Each year in individual plot or (False) all years collected in one plot
    "duration_curve": False,                # True: Plot duration curve, or False: Plot storage filling over time
    "save_fig": False,                      # True: Save plot as pdf
    "interval": 1,                          # Number of months on x-axis. 1 = Step is one month, 12 = Step is 12 months
    "tex_font": False                       # Keep false
}

# === COMPUTE TIMERANGE AND PLOT FLOW ===
time_ZP = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
calcPlot_ZonalPrices_FromDB(data, database, time_ZP, OUTPUT_PATH_PLOTS, DATE_START, plot_config)



# %% Check Total Consumption for a given period.
# Demand Response
# === INITIALIZATIONS ===
START = {"year": 2020, "month": 1, "day": 1, "hour": 0}
END = {"year": 2020, "month": 12, "day": 31, "hour": 23}
area = 'NO'

time_Demand = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
demandTotal = getDemandPerAreaFromDB(data, database, area=area, timeMaxMin=time_Demand)
print(sum(demandTotal['sum']))


# %% === Get Production Data ===
# === INITIALIZATIONS ===
START = {"year": 2020, "month": 1, "day": 1, "hour": 0}
END = {"year": 2020, "month": 12, "day": 31, "hour": 23}
area = 'PL'

time_Prod = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
total_Production = getProductionPerAreaFromDB(data, database, time_Prod, area)
print(total_Production)


# %% === CHECK SPILLED VS PRODUCED ===
START = {"year": 1992, "month": 12, "day": 1, "hour": 0}
END = {"year": 1992, "month": 12, "day": 2, "hour": 23}
time_EB = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
gen_idx = [381]
sum_spilled, sum_produced = checkSpilled_vs_ProducedAtGen(database, gen_idx, time_EB)

# TODO: Make to check within a zone for all generators of the same type

# %% === GET IMPORTS/EXPORTS FOR EACH ZONE ===

START = {"year": 1991, "month": 1, "day": 1, "hour": 0}
END = {"year": 2020, "month": 12, "day": 31, "hour": 23}
time_EB = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
# flow_data = getFlowDataOnALLBranches(data, database, time_EB)
flow_data = collectFlowDataOnALLBranches(data, database, time_EB)

zone_imports, zone_exports = getZoneImportExports(data, flow_data)
# Example: Print results
print("Zone Imports (importer, exporter): Total Import [MWh]")
for (importer, exporter), total in zone_imports.items():
    print(f"{importer} importing from {exporter}: {total:.2f} MWh")

"""
Colormap options:
- 'YlOrRd': Yellow to Red
- 'Blues': Blue shades
- 'Greens': Green shades
- 'Purples': Purple shades
- 'Oranges': Orange shades
- 'Greys': Grey shades
- 'viridis': Viridis colormap
- 'plasma': Plasma colormap
- 'cividis': Cividis colormap
- 'magma': Magma colormap
- 'copper': Copper colormap
- 'coolwarm': Coolwarm colormap
- 'RdBu': Red to Blue colormap
- 'Spectral': Spectral colormap
- 'twilight': Twilight colormap
- 'twilight_shifted': Twilight shifted colormap
- 'cubehelix': Cubehelix colormap
- 'terrain': Terrain colormap
- 'ocean': Ocean colormap
- 'RdYlGn': Red to Yellow to Green colormap
"""

# %% === GET IMPORTS/EXPORTS FOR EACH ZONE ===

START = {"year": 1991, "month": 1, "day": 1, "hour": 0}
END = {"year": 2020, "month": 12, "day": 31, "hour": 23}
time_EB = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
# flow_data = getFlowDataOnALLBranches(data, database, time_EB)
flow_data = collectFlowDataOnALLBranches(data, database, time_EB)

zone_imports, zone_exports = getZoneImportExports(data, flow_data)
# Example: Print results
print("Zone Imports (importer, exporter): Total Import [MWh]")
for (importer, exporter), total in zone_imports.items():
    print(f"{importer} importing from {exporter}: {total:.2f} MWh")


# %% === Get flow flow between zones

START = {"year": 1991, "month": 1, "day": 1, "hour": 0}
END = {"year": 1991, "month": 12, "day": 31, "hour": 23}
time_EB = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)


df_zone_flows = getFlowBetweenAllZones(data, database, time_EB)

# Use it in the DataFrame
print(df_zone_flows.columns)
# Save to CSV
df_zone_flows.to_csv(OUTPUT_PATH / 'zone_to_zone_flows.csv')



# %% CHECK GENERATION TO SENSITIVITY

# === INITIALIZATIONS ===
GEN_TYPE = 'wind_off'
START = {"year": 1991, "month": 1, "day": 1, "hour": 0}
END = {"year": 1991, "month": 12, "day": 31, "hour": 23}
time_period = get_hour_range(SIM_YEAR_START, SIM_YEAR_END, TIMEZONE, START, END)
min_time, max_time = time_period  # Unpack min, max from time_period

# Filter generators by type
generator_idx = data.generator[data.generator.type == GEN_TYPE].index.tolist()

# Get inflow ref to generator
generator_inflow = data.generator.inflow_ref[generator_idx].tolist()
gen_to_inflow_map = dict(zip(generator_idx, generator_inflow))

# Get inflow profile to generator
generator_inflow_profile = data.profiles[data.generator.inflow_ref[generator_idx].unique().tolist()].loc[
                           min_time:max_time]

node_ids = data.node[data.node.id.isin(data.generator.node[generator_idx].unique())].index.tolist()
# %%
# Retrieve and process sensitivity data
print("Get Sensitivity Data")
df_sens = database.getResultGeneratorSens(time_period, generator_idx)


print("Get Nodal Prices")
df_node_price = database.getResultNodalPricesPerNode(node_ids, time_period)
df_node_price = pd.DataFrame(df_node_price)


print("Get Generator Data")
raw_gen_data = database.getAllGeneratorPowerTest(time_period, generator_idx)
df_gen = pd.DataFrame(raw_gen_data, columns=["time", "gen_id", "output"])
df_gen = df_gen.pivot(index="time", columns="gen_id", values="output")

# %%
node_idx = df_node_price.columns.tolist()
node_ids = data.node.loc[node_idx, "id"].tolist()

index_to_node_id = dict(zip(node_idx, node_ids))
node_id_to_index = dict(zip(node_ids, node_idx))

# Generator to node ID mapping
gen_to_node = data.generator.node[generator_idx].to_dict()


# %%


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset



def plot_gen_vs_sens_and_price(df_gen, df_sens, df_node_price, gen_to_node, node_id_to_index, generator_ids=None):
    if generator_ids is None:
        generator_ids = df_gen.columns.tolist()

    for gen_id in generator_ids:
        if gen_id not in df_gen.columns or gen_id not in df_sens.columns:
            continue

        node_id = gen_to_node.get(gen_id)
        node_idx = node_id_to_index.get(node_id)

        if node_idx not in df_node_price.columns:
            print(f"Skipping generator {gen_id}: no node price for node ID {node_id} (index {node_idx})")
            continue

        gen = df_gen[gen_id]
        sens = df_sens[gen_id]
        price = df_node_price[node_idx]

        df_plot = pd.DataFrame({
            "Generation": gen,
            "Sensitivity": sens,
            "Nodal Price": price
        })

        fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=False)

        # Plot Generation
        axs[0].plot(df_plot.index, df_plot["Generation"], color='green')
        axs[0].set_ylabel("Generation (MW)")
        axs[0].set_title(f"Gen {gen_id}: Generation over Time")

        # Plot Sensitivity
        axs[1].plot(df_plot.index, df_plot["Sensitivity"], color='blue')
        axs[1].set_ylabel("Sensitivity")
        axs[1].set_title("Sensitivity over Time")

        # Plot Nodal Price
        axs[2].plot(df_plot.index, df_plot["Nodal Price"], color='purple')
        axs[2].set_ylabel("Nodal Price (€/MWh)")
        axs[2].set_title("Nodal Price over Time")
        axs[2].set_xlabel("Time")

        plt.tight_layout()
        plt.show()




plot_gen_vs_sens_and_price(df_gen, df_sens, df_node_price, gen_to_node, node_id_to_index, generator_ids=[383, 385, 391])

# %%
# Filter to generator of interest (e.g., 383)
gen_id = 391
df_gen_filtered = df_gen[gen_id].copy()

# Make sure time is datetime (optional)
# df_gen_filtered["Time"] = pd.to_datetime(df_gen_filtered["Time"])
# df_gen_filtered.set_index("Time", inplace=True)

# Plot
fig, ax = plt.subplots(figsize=(12, 6))

# Main plot
ax.plot(df_gen_filtered.index, df_gen_filtered, color="green")
ax.set_title(f"Generator {gen_id}: Generation over Time with Zoom")
ax.set_ylabel("Generation (MW)")
ax.set_xlabel("Time")

# Inset: zoom in on 3900–3924 hours (assuming index is hourly datetime or integers)
# If index is datetime, adjust the condition to fit actual timestamps.
df_zoom = df_gen_filtered.iloc[3900:3925]  # 3900 to 3924 inclusive

# Inset axes
axins = inset_axes(ax, width="30%", height="30%", loc="upper right")
axins.plot(df_zoom.index, df_zoom, color="green")
axins.set_xlim(df_zoom.index[0], df_zoom.index[-1])
axins.set_ylim(df_zoom.min(), df_zoom.max())
# axins.set_xticklabels([])
# axins.set_yticklabels([])
axins.set_xlim(3900, 3925)
axins.set_ylim(50, 300)
axins.grid(True, linestyle='--', alpha=0.5)
# Tegn linjer mellom zoom og hovedfigur#
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="black", lw=1)
# Connect inset to main plot
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

plt.tight_layout()
plt.show()
