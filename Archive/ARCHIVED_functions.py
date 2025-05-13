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
