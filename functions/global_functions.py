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


def read_grid_data(year, date_start, date_end, data_path):
    """
    Reads and processes grid data for a specified year and date range.

    Parameters:
        year (int): The year for which data should be loaded.
        date_start (str): The start date of the simulation period in 'YYYY-MM-DD' format.
        date_end (str): The end date of the simulation period in 'YYYY-MM-DD' format.
        data_path (str or pathlib.Path): The base path to the data directory.

    Returns:
        powergama.GridData: An instance of GridData containing the processed grid data.
    """
    # Calculate and print the number of simulation hours and years
    datapath_GridData = data_path/ "system"
    file_storval_filling = pathlib.Path(data_path / "storage/profiles_storval_filling.csv")
    file_30y_profiles = pathlib.Path(data_path / "timeseries_profiles.csv")

    # Initialize GridData object
    data = powergama.GridData()
    data.readGridData(nodes=datapath_GridData / "node.csv",
                      ac_branches=datapath_GridData / "branch.csv",
                      dc_branches=datapath_GridData / "dcbranch.csv",
                      generators=datapath_GridData / "generator.csv",
                      consumers=datapath_GridData / "consumer.csv")

    # Read and process 30-year profiles
    profiles_30y = pd.read_csv(file_30y_profiles, index_col=0, parse_dates=True)
    profiles_30y["const"] = 1
    data.profiles = profiles_30y[(profiles_30y.index >= date_start) & (profiles_30y.index < date_end)].reset_index()
    data.storagevalue_time = data.profiles[["const"]]

    # Read storage value filling data
    storval_filling = pd.read_csv(file_storval_filling)
    data.storagevalue_filling = storval_filling

    # Set the timerange and time delta for the simulation
    data.timerange = list(range(data.profiles.shape[0]))
    data.timeDelta = 1.0  # hourly data

    # Calculate and print the number of simulation hours and years
    num_hours = data.timerange[-1] - data.timerange[0]
    print(f'Simulation hours: {num_hours}')
    num_years = num_hours / (365.2425 * 24)
    print(f'Simulation years: {np.round(num_years, 3)}')

    # Filter offshore wind farms by year:
    data.generator = data.generator[~(data.generator["year"] > year)].reset_index(drop=True)

    # remove zero capacity generators:
    m_gen_hascap = data.generator["pmax"] > 0
    data.generator = data.generator[m_gen_hascap].reset_index(drop=True)

    return data


# Read and configure grid
def setup_grid(year, date_start, date_end, data_path, new_scenario, save_scenario):
    """
    Set up grid data and initialize a simulation scenario.

    This function reads grid data for the specified year and date range,
    then configures the base grid data with a specific scenario file.

    Parameters:
        year (int): The year for which data should be loaded.
        date_start (str): The start date of the simulation period.
        date_end (str): The end date of the simulation period.

    Returns:
        data (Scenario): Configured grid data for simulation.
        time_max_min (list): List containing the start and end indices for the simulation timeframe.
    """
    data = read_grid_data(year, date_start, date_end, data_path)
    time_max_min = [0, len(data.timerange)]
    scenario_file = pathlib.Path(data_path / f"scenario_{year}.csv")
    if new_scenario:
        data = pgs.newScenario(base_grid_data=data, scenario_file=scenario_file)

    save_scenario_file = pathlib.Path(data_path / f"current_scenario_{year}.csv")

    if save_scenario:
        pgs.saveScenario(base_grid_data=data, scenario_file=save_scenario_file)

    return data, time_max_min



def solve_lp(data, sql_file, loss_method, replace):
    """
    Solves a linear programming problem using the given grid data and stores the results in a SQL file.

    Parameters:
        data (powergama.GridData): The grid data to be used for the linear programming problem.
        sql_file (str): The path to the SQL file where the results will be stored.

    Returns:
        powergama.Results: The results of the linear programming problem.
    """

    lp = powergama.LpProblem(grid=data, lossmethod=loss_method)  # lossmethod; 0=no losses, 1=linearised losses, 2=added as load
    # if replace = False, bruker kun sql_file som input
    res = powergama.Results(data, sql_file, replace=replace)
    if replace:
        start_time = time.time()
        lp.solve(res, solver="glpk")
        end_time = time.time()
        print("\nSimulation time = {:.2f} seconds".format(end_time - start_time))
        print("\nSimulation time = {:.2f} minutes".format((end_time - start_time)/60))
    return res



"""Plot Functions"""
def configure_axes(ax, relative, x_label):
    """
    Configures the y-axes for a plot based on the given parameters.

    Parameters:
        ax (matplotlib.axes.Axes): The axes object to configure.
        relative (bool): If True, set the y-axis label to 'Reservoir Filling [%]'.
                         If False, set the y-axis label to 'Reservoir Filling [TWh]'.
        x_label (str): The label for the x-axis.
    Returns:
        None
    """
    ax.set_xlabel(x_label)
    if relative:
        ax.set_ylabel('Reservoir Filling [%]')
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x}%'))

    else:
        ax.set_ylabel('Reservoir Filling [TWh]')
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x / 1e6:.1f}'))


def plot_storage_filling_area(storfilling, DATE_START, DATE_END, areas, interval, title, OUTPUT_PATH_PLOTS,
                              relative, plot_by_year, save_plot, duration_curve, tex_font):
    """
    Plots the storage filling levels for specified areas over a given date range.

    Parameters:
        storfilling (pd.DataFrame): DataFrame containing storage filling data.
        DATE_START (str): Start date of the plot in 'YYYY-MM-DD' format.
        DATE_END (str): End date of the plot in 'YYYY-MM-DD' format.
        areas (list): List of areas to plot.
        interval (int): Interval for x-axis ticks.
        title (str): Title of the plot.
        OUTPUT_PATH_PLOTS (pathlib.Path): Path to save the plot.
        relative (bool): If True, plot relative storage filling percentages.
        plot_by_year (bool): If True, plot data by day of the year.
        save_plot (bool): If True, save the plot to the specified path.
        duration_curve (bool): If True, plot a duration curve instead of standard

    Raises:
        ValueError: If an area is not found in the storfilling DataFrame columns.
    """
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

        if relative:
            ax.set_yticks([0, 20, 40, 60, 80, 100])
            ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
            ax.set_ylim(0, 100)

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

    plt.title(title)
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


def plot_nodal_prices(res, node_prices, nodes_in_zone, zone, DATE_START, DATE_END, interval, TITLE, plot_all_nodes,
                      save_plot_nodal, OUTPUT_PATH_PLOTS, plot_by_year_nodal, duration_curve_nodal, tex_font):
    if plot_all_nodes:
        # Plotting the converted nodal prices for the NO2 nodes
        fig, ax1 = plt.subplots(figsize=(10, 6))
        if not duration_curve_nodal:
            for node in nodes_in_zone:
                ax1.plot(node_prices.index, node_prices[node],
                         label=f"Node {res.grid.node.loc[node, 'id']}")

            ax1.set_xlabel('Date')
            ax1.set_ylabel('Price (EUR/MWh)')
            # Customize x-axis to show ticks every second month
            ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

            # Set x-axis limits to cover the entire data range
            ax1.set_xlim(pd.to_datetime(DATE_START), pd.to_datetime(DATE_END))

            lines1, labels1 = ax1.get_legend_handles_labels()
            ax1.legend(lines1, labels1, loc='upper right')  # , bbox_to_anchor=(1, 0))
        else:
            for node in nodes_in_zone:
                sorted_values = node_prices[node].sort_values(ascending=False).reset_index(drop=True)
                ax1.plot(sorted_values, label=f"Node {res.grid.node.loc[node, 'id']} (Duration Curve)")

            ax1.set_xlabel('Rank')
            ax1.set_ylabel('Price (EUR/MWh)')

            lines1, labels1 = ax1.get_legend_handles_labels()
            ax1.legend(lines1, labels1, loc='upper right')

    else:
        # Average nodal prices for the zone
        avg_node_prices = pd.DataFrame((node_prices.sum(axis=1) / len(nodes_in_zone)), columns=['avg_price'])
        avg_node_prices['year'] = avg_node_prices.index.year  # Add year column to DataFrame
        # temp_nodes_in_zone = nodes_in_zone[0]

        fig, ax1 = plt.subplots(figsize=(10, 6))
        if not plot_by_year_nodal:
            if not duration_curve_nodal:
                ax1.plot(avg_node_prices.index, avg_node_prices['avg_price'], label='Average Zone Price')
                ax1.set_xlabel('Date')
                ax1.set_ylabel('Price (EUR/MWh)')
                # Customize x-axis to show ticks every second month
                ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
                plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
                ax1.set_xlim(pd.to_datetime(DATE_START), pd.to_datetime(DATE_END))
            else:
                sorted_values = avg_node_prices['avg_price'].sort_values(ascending=False).reset_index(drop=True)
                ax1.plot(sorted_values, label='Average Zone Price (Duration Curve)')
                ax1.set_xlabel('Rank')
                ax1.set_ylabel('Price (EUR/MWh)')
        else:
            for year in avg_node_prices['year'].unique():
                group = avg_node_prices[avg_node_prices['year'] == year]
                if duration_curve_nodal:
                    sorted_values = group['avg_price'].sort_values(ascending=False).reset_index(drop=True)
                    ax1.plot(sorted_values, label=f"{year}")
                    ax1.set_xlabel('Hour')
                    ax1.set_ylabel('Price (EUR/MWh)')
                else:
                    ax1.plot(group.index.dayofyear, group['avg_price'], label=f"{year}")
                    ax1.set_xlabel('Date')
                    ax1.set_ylabel('Price (EUR/MWh)')

            if not duration_curve_nodal:
                # Customize x-axis to show ticks every month
                ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
                plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
                ax1.set_xlim(0, 365)

        lines1, labels1 = ax1.get_legend_handles_labels()
        ax1.legend(lines1, labels1, loc='upper right')  # , bbox_to_anchor=(1, 0))

    plt.title(TITLE)
    plt.grid(True)
    plt.tight_layout()
    if tex_font:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"]})
    if save_plot_nodal:
        plt.savefig(OUTPUT_PATH_PLOTS / f'nodal_price_{zone}.pdf')
    plt.show()



def calculate_Hydro_Res_Inflow(res, data, DATE_START, area_OP, genType, time_max_min, include_pump):
    """
    Calculate hydro reservoir inflow, production, and storage filling.

    Parameters:
        res (powergama.Results): The results object containing simulation data.
        data (powergama.Scenario): The scenario data containing profiles and grid information.
        DATE_START (str): The start date of the simulation period in 'YYYY-MM-DD' format.
        area_OP (str): The operational area for which to calculate inflow.
        genType (str): The type of generator (e.g., 'hydro').
        time_max_min (list): List containing the start and end indices for the simulation timeframe.
        include_pump (bool): If True, include pump power in the calculations.

    Returns:
        pd.DataFrame: A DataFrame containing resampled data for reservoir filling, hydro production, and inflow.
    """
    genTypeIdx = res.grid.getGeneratorsPerAreaAndType()[area_OP][genType]
    # Inflow
    inflowFactor = res.grid.generator["inflow_fac"][
        genTypeIdx[0]]  # Assuming all generators have the same inflow factor
    inflowProfile = res.grid.generator["inflow_ref"][
        genTypeIdx]  # All generators have different inflow profiles, zone-based
    inflow_df = pd.DataFrame()

    for gen in genTypeIdx:
        prodCap = res.grid.generator['pmax'][gen]  # MW
        inflow_value = data.profiles[inflowProfile[gen]]
        inflow_df[gen] = [i * prodCap * inflowFactor for i in inflow_value]
    inflow = inflow_df.sum(axis=1)
    print(f"Total inflow: {sum(inflow) / 1e6:.2f} TWh")

    # Hydro production
    hydro_production_sum = pd.DataFrame(res.db.getResultGeneratorPower(genTypeIdx, time_max_min)).sum(axis=1)

    # Storage filling
    storage_filling = res.getStorageFillingInAreas(areas=['NO'], generator_type="hydro")
    storage_filling_percentage = [value * 100 for value in storage_filling]

    if include_pump:
        # Pump Power, it seems all generators have zero pump output
        pumpOutput = []
        for gen in genTypeIdx:
            # if res.grid.generator["pump_cap"][gen] > 0:
            pumpOutput.append(res.db.getResultPumpPower(gen, time_max_min))

    # Create DataFrame
    df = pd.DataFrame({
        'Reservoir Filling': storage_filling_percentage,
        'Hydro Production': hydro_production_sum,
        'Inflow': inflow
    })
    df.index = pd.date_range(DATE_START, periods=time_max_min[-1], freq='h')  # Set index to date range
    print(f"Total Hydro production: {hydro_production_sum.sum() / 1e6:.2f} TWh")
    # Resample the data
    df_resampled = df.resample('7D').agg({
        'Reservoir Filling': 'last',
        'Hydro Production': 'sum',
        'Inflow': 'sum'
    })
    return df_resampled


def plot_hydro_prod_res_inflow(df, DATE_START, DATE_END, interval, TITLE, OUTPUT_PATH_PLOTS, save_plot, box_in_frame,
                               plot_full_timeline, tex_font):
    """
    Plots hydro production, inflow, and reservoir filling over a specified date range.

    Parameters:
        df (pd.DataFrame): DataFrame containing hydro production, inflow, and reservoir filling data.
        DATE_START (str): Start date of the plot in 'YYYY-MM-DD' format.
        DATE_END (str): End date of the plot in 'YYYY-MM-DD' format.
        interval (int): Interval for x-axis ticks.
        TITLE (str): Title of the plot.
        OUTPUT_PATH_PLOTS (pathlib.Path): Path to save the plot.
        save_plot (bool): If True, save the plot to the specified path.
        box_in_frame (bool): If True, place the legend inside the plot frame.
        plot_full_timeline (bool): If True, plot the full timeline; otherwise, plot by year.

    Returns:
        None
    """
    if plot_full_timeline:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        # Plot Hydro Production
        ax1.plot(df.index, df['Hydro Production'], color='b', label='Hydro Production')
        ax1.plot(df.index, df['Inflow'], color='g', label='Inflow')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Hydro Production / Inflow (TWh/week)', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x / 1e6:.1f}'))

        # Create a twin axis for Reservoir Filling
        ax2 = ax1.twinx()
        ax2.plot(df.index, df['Reservoir Filling'], color='k', label='Reservoir Filling')
        ax2.set_ylabel('Reservoir Filling (%)', color='k')
        ax2.tick_params(axis='y', labelcolor='k')
        # Customize y-axis for reservoir
        ax2.set_yticks([0, 20, 40, 60, 80, 100])
        ax2.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
        ax2.set_ylim(0, 100)

        # Customize x-axis to show ticks every second month
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

        # Set x-axis limits to cover the entire data range
        ax1.set_xlim(pd.to_datetime(DATE_START), pd.to_datetime(DATE_END))

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        if box_in_frame:
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        else:
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='lower left', bbox_to_anchor=(1, 0))

        plt.title(TITLE + f'for weather years in period {DATE_START[0:4]}-{DATE_END[0:4]}')
    else:
        for year in df['year'].unique():
            year_data = df[df['year'] == year]
            fig, ax1 = plt.subplots(figsize=(10, 6))

            # Plot Hydro Production
            ax1.plot(year_data.index, year_data['Hydro Production'], color='b', label='Hydro Production')
            ax1.plot(year_data.index, year_data['Inflow'], color='g', label='Inflow')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Hydro Production / Inflow (TWh/week)', color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x / 1e6:.1f}'))

            # Create a twin axis for Reservoir Filling
            ax2 = ax1.twinx()
            ax2.plot(year_data.index, year_data['Reservoir Filling'], color='k', label='Reservoir Filling')
            ax2.set_ylabel('Reservoir Filling (%)', color='k')
            ax2.tick_params(axis='y', labelcolor='k')

            # Customize y-axis for reservoir
            ax2.set_yticks([0, 20, 40, 60, 80, 100])
            ax2.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
            ax2.set_ylim(0, 100)

            # Customize x-axis to show ticks every second month
            ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

            # Set x-axis limits to cover the data range for the current year
            start_of_year = pd.Timestamp(f'{year}-01-01')
            end_of_year = pd.Timestamp(f'{year}-12-31')
            ax1.set_xlim(start_of_year, end_of_year)

            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            if box_in_frame:
                ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            else:
                ax2.legend(lines1 + lines2, labels1 + labels2, loc='lower left', bbox_to_anchor=(1, 0))

            plt.title(TITLE + f' for weather year {year}')

    plt.grid(True)
    plt.tight_layout()
    if tex_font:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"]})
    if save_plot:
        plt.savefig(OUTPUT_PATH_PLOTS / 'hydro_production.pdf')
    plt.show()


def calc_PLP(res, area_OP, DATE_START, time_max_min):
    nodes_in_zone_2 = res.grid.node[res.grid.node['area'] == area_OP].index.tolist()
    # Get nodal prices for all nodes in the zone in one step and apply conversion factor
    node_prices_2 = pd.DataFrame({node: res.getNodalPrices(node=node) for node in nodes_in_zone_2})
    node_prices_2.index = pd.date_range(DATE_START, periods=time_max_min[-1], freq='h')
    avg_node_prices_2 = node_prices_2.sum(axis=1) / len(nodes_in_zone_2)

    # Load demand
    load_demand = res.getDemandPerArea(area='NO')

    genHydroIdx = res.grid.getGeneratorsPerAreaAndType()[area_OP]['hydro']
    hydro_production_sum = pd.DataFrame(res.db.getResultGeneratorPower(genHydroIdx, time_max_min)).sum(axis=1)
    hydro_production_sum.index = pd.date_range(DATE_START, periods=time_max_min[-1], freq='h')

    df_plp = pd.DataFrame({
        'Production': hydro_production_sum,
        'Load': load_demand['sum'],
        'Price': avg_node_prices_2
    })

    df_plp.index = pd.date_range(DATE_START, periods=time_max_min[-1], freq='h')

    df_plp_resampled = df_plp.resample('7D').agg({
        'Production': 'sum',
        'Load': 'sum',
        'Price': 'mean'
    })

    df_plp['year'] = df_plp.index.year
    df_plp_resampled['year'] = df_plp_resampled.index.year

    return df_plp, df_plp_resampled


def plot_hydro_prod_demand_price(df_plp, df_plp_resampled, resample, DATE_START, DATE_END, interval, TITLE, save_fig,
                                 plot_full_timeline, box_in_frame, OUTPUT_PATH_PLOTS, tex_font):
    if resample:
        df = df_plp_resampled
    else:
        df = df_plp

    if plot_full_timeline:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(df.index, df['Production'], label='Hydro Production', color='blue')
        ax1.plot(df.index, df['Load'], label='Load', color='red')
        ax1.set_xlabel('Date')
        if resample:
            ax1.set_ylabel('Production/Load (TWh)')
            ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x / 1e6:.1f}'))
        else:
            ax1.set_ylabel('Production/Load (MWh)')
            ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.1f}'))
        ax2 = ax1.twinx()
        ax2.plot(df.index, df['Price'], label='Price', color='orange')
        ax2.set_ylabel('Price (EUR/MWh)')

        # Customize x-axis to show ticks every second month
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

        # Set x-axis limits to cover the entire data range
        ax1.set_xlim(pd.to_datetime(DATE_START), pd.to_datetime(DATE_END))

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        if box_in_frame:
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        else:
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='lower left', bbox_to_anchor=(1, 0))
        plt.title(TITLE)
    else:
        for year in df['year'].unique():
            year_data = df[df['year'] == year]
            if year_data.empty:
                print(f"No data available for year {year}")
                continue

            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax1.plot(year_data.index, year_data['Production'], label='Hydro Production', color='blue')
            ax1.plot(year_data.index, year_data['Load'], label='Load', color='red')

            ax1.set_xlabel('Month')
            if resample:
                ax1.set_ylabel('Production/Load (TWh)')
                ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x / 1e6:.1f}'))
            else:
                ax1.set_ylabel('Production/Load (MWh)')
                ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.1f}'))

            ax2 = ax1.twinx()
            ax2.plot(year_data.index, year_data['Price'], label='Price', color='orange')
            ax2.set_ylabel('Price (EUR/MWh)')

            # Customize x-axis to show ticks every second month
            ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

            # Set x-axis limits to cover the data range for the current year
            start_of_year = pd.Timestamp(f'{year}-01-01')
            end_of_year = pd.Timestamp(f'{year}-12-30')
            ax1.set_xlim(start_of_year, end_of_year)

            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            if box_in_frame:
                ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            else:
                ax2.legend(lines1 + lines2, labels1 + labels2, loc='lower left', bbox_to_anchor=(1, 0))

            plt.title(TITLE + f' for weather year {year}')

    plt.grid(True)
    plt.tight_layout()
    if tex_font:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"]})
    if save_fig:
        plt.savefig(OUTPUT_PATH_PLOTS / 'nodal_price_demand_hydro_production.pdf')
    plt.show()


def check_load_shedding(load_shedding, tex_font):
    if load_shedding.sum() == 0:
        print("No load shedding in the given year")
    else:
        print(f"Total load shedding in the given year: {load_shedding.sum()} MWh")

        # Plot shedding as bar
        plt.figure(figsize=(10, 6))
        load_shedding.plot(kind='bar')
        plt.xlabel("Node")
        plt.ylabel("Load Shedding (MWh)")
        plt.title("Load Shedding per Node in 2018")
        plt.grid(True)
        plt.tight_layout()
        if tex_font:
            plt.rcParams.update({
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Computer Modern Roman"]})
        plt.show()



def get_production_by_type(res, area_OP, time_max_min, DATE_START):

    # Get Generation by type
    genHydro = 'hydro'
    genHydroIdx = res.grid.getGeneratorsPerAreaAndType()[area_OP][genHydro]
    all_hydro_production = pd.DataFrame(res.db.getResultGeneratorPower(genHydroIdx, time_max_min)).sum(axis=1)

    genWind = 'wind_on'
    genWindIdx = res.grid.getGeneratorsPerAreaAndType()[area_OP][genWind]
    all_wind_production = pd.DataFrame(res.db.getResultGeneratorPower(genWindIdx, time_max_min)).sum(axis=1)

    genSolar = 'solar'
    genSolarIdx = res.grid.getGeneratorsPerAreaAndType()[area_OP][genSolar]
    all_solar_production = pd.DataFrame(res.db.getResultGeneratorPower(genSolarIdx, time_max_min)).sum(axis=1)

    genGas = 'fossil_gas'
    genGasIdx = res.grid.getGeneratorsPerAreaAndType()[area_OP][genGas]
    all_gas_production = pd.DataFrame(res.db.getResultGeneratorPower(genGasIdx, time_max_min)).sum(axis=1)

    # Get Load Demand
    load_demand = res.getDemandPerArea(area='NO')

    # Get Avg Price for Area
    nodes_in_area = res.grid.node[res.grid.node['area'] == area_OP].index.tolist()
    node_prices_3 = pd.DataFrame({node: res.getNodalPrices(node=node) for node in nodes_in_area})
    node_prices_3.index = pd.date_range(DATE_START, periods=time_max_min[-1], freq='h')
    avg_area_prices = node_prices_3.sum(axis=1) / len(nodes_in_area)

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



def plot_production(df_gen_resampled, df_prices_resampled, DATE_START, DATE_END, interval, fig_size, TITLE,
                     OUTPUT_PATH_PLOTS, plot_full_timeline, plot_duration_curve, save_fig, box_in_frame, tex_font):
    if plot_full_timeline:
        # Full timeline plot
        fig, ax1 = plt.subplots(figsize=(10, 6))
        if plot_duration_curve:
            for prod in df_gen_resampled.columns:
                sorted_values = df_gen_resampled[prod].sort_values(ascending=False).reset_index(drop=True)
                ax1.plot(sorted_values, label=f"{prod} (Duration Curve)")
            ax1.set_xlabel('Rank')
            ax1.set_ylabel('Power [MW]')
        else:
            for prod in df_gen_resampled.columns:
                ax1.plot(df_gen_resampled.index, df_gen_resampled[prod], label=f"{prod}")
            ax1.set_xlabel('Time (hours)')
            ax1.set_ylabel('Power [MW]')

        ax2 = ax1.twinx()
        if not plot_duration_curve:
            ax2.plot(df_prices_resampled.index, df_prices_resampled['Price'], color='black', label='Price (right)')
            ax2.set_ylabel('Price [EUR/MWh]')

        # Customize x-axis
        if not plot_duration_curve:
            ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

            ax1.set_xlim(pd.to_datetime(DATE_START), pd.to_datetime(DATE_END))

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels() if not plot_duration_curve else ([], [])
        if box_in_frame:
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        else:
            ax1.legend(lines1 + lines2, labels1 + labels2,
                       loc='upper center', bbox_to_anchor=(0.5, -0.2),
                       ncol=3, frameon=False)  # Adjust ncol to fit the number of legend items

        plt.title(TITLE)
        plt.grid(True)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)  # Make space for the legend under the graph
        if tex_font:
            plt.rcParams.update({
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Computer Modern Roman"]})
        if save_fig:
            plt.savefig(OUTPUT_PATH_PLOTS / 'production_prices_full_timeline.pdf')
        plt.show()

    else:
        # Plot separately for each year
        df_gen_resampled['year'] = df_gen_resampled.index.year
        df_prices_resampled['year'] = df_prices_resampled.index.year

        for year in df_gen_resampled['year'].unique():
            year_data_gen = df_gen_resampled[df_gen_resampled['year'] == year].drop(columns='year')
            year_data_prices = df_prices_resampled[df_prices_resampled['year'] == year].drop(columns='year')

            fig, ax1 = plt.subplots(figsize=fig_size)
            if plot_duration_curve:
                for prod in year_data_gen.columns:
                    sorted_values = year_data_gen[prod].sort_values(ascending=False).reset_index(drop=True)
                    ax1.plot(sorted_values, label=f"{prod} (Duration Curve)")
                ax1.set_xlabel('Time (hours)')
                ax1.set_ylabel('Produced Power / Load [MWh]')
            else:
                for prod in year_data_gen.columns:
                    ax1.plot(year_data_gen.index, year_data_gen[prod], label=f"{prod}")
                ax1.set_xlabel('Time (hours)')
                ax1.set_ylabel('Produced Power / Load [MWh]')

            ax2 = ax1.twinx()
            if not plot_duration_curve:
                ax2.plot(year_data_prices.index, year_data_prices['Price'], color='black', label='Price (right)')
                ax2.set_ylabel('Price [EUR/MWh]')

            # Customize x-axis
            if not plot_duration_curve:
                ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
                plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

                start_of_year = pd.Timestamp(f'{year}-01-01')
                end_of_year = pd.Timestamp(f'{year}-12-31')
                ax1.set_xlim(start_of_year, end_of_year)

            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels() if not plot_duration_curve else ([], [])
            if box_in_frame:
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            else:
                ax1.legend(lines1 + lines2, labels1 + labels2,
                           loc='upper center', bbox_to_anchor=(0.5, -0.2),
                           ncol=3, frameon=False)  # Adjust ncol to fit the number of legend items

            plt.title(TITLE + f' for Year {year}' + (' (Duration Curve)' if plot_duration_curve else ''))
            plt.grid(True)
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.25)  # Make space for the legend under the graph
            if tex_font:
                plt.rcParams.update({
                    "text.usetex": True,
                    "font.family": "serif",
                    "font.serif": ["Computer Modern Roman"]})
            if save_fig:
                plt.savefig(
                    OUTPUT_PATH_PLOTS / f'production_prices_{year}{"_duration_curve" if plot_duration_curve else ""}.pdf')
            plt.show()



###############################################################################################################

"""Flow Based Functions"""
def create_price_and_utilization_map(data, res, time_max_min, output_path, eur_to_nok):
    """
    Generate a folium map displaying nodal prices and branch utilization.

    This function creates a map with nodes representing average prices and branches representing
    line utilization for the given time range. The map is saved as an HTML file.

    Parameters:
        data (Scenario):        Simulation data containing nodes and branches.
        res (Result):           Result object with nodal prices, utilization, and flows.
        time_max_min (list):    List specifying the start and end time steps for the simulation.
        output_path (str):      Path where the HTML map file will be saved.
        eur_to_nok (float):     Conversion rate from EUR to NOK for price display.

    Returns:
        None
    """
    nodal_prices = res.getAverageNodalPrices(time_max_min)
    avg_area_price = res.getAreaPricesAverage(timeMaxMin=time_max_min)
    ac_utilisation = res.getAverageUtilisation(time_max_min, branchtype="ac")
    dc_utilisation = res.getAverageUtilisation(time_max_min, branchtype="dc")
    ac_flows = res.getAverageBranchFlows(time_max_min, branchtype="ac")
    dc_flows = res.getAverageBranchFlows(time_max_min, branchtype="dc")

    f = folium.Figure(width=700, height=800)
    m = folium.Map(location=[data.node["lat"].mean(), data.node["lon"].mean()], zoom_start=4.4)
    m.add_to(f)

    colormap = cm.LinearColormap(['green', 'yellow', 'red'], vmin=min(nodal_prices), vmax=max(nodal_prices))
    colormap.caption = 'Nodal Prices'
    colormap.add_to(m)

    for i, price in enumerate(nodal_prices):
        add_node_marker(data, i, price, avg_area_price, m, colormap, eur_to_nok)

    line_colormap = cm.LinearColormap(['green', 'yellow', 'red'], vmin=0, vmax=1)
    line_colormap.caption = 'Branch Utilisation'
    line_colormap.add_to(m)

    add_branch_lines(data, ac_utilisation, ac_flows[2], 'AC', m, line_colormap)
    add_branch_lines(data, dc_utilisation, dc_flows[2], 'DC', m, line_colormap, dashed=True)

    m.save(output_path)
    display(m)


def add_node_marker(data, index, price, avg_national_prices, m, colormap, eur_to_nok):
    """
    Add a marker to the map for a specific node, displaying its price and zone information.

    Parameters:
        data (Scenario):            Simulation data containing node information.
        index (int):                Node index within the data.
        price (float):              Nodal price for the given node.
        avg_national_prices (dict): Dictionary of average national prices by area.
        m (folium.Map):             Folium map object to which the marker will be added.
        colormap (LinearColormap):  Colormap for representing nodal prices.
        eur_to_nok (float):         Conversion rate from EUR to NOK.

    Returns:
        None
    """
    lat = data.node.loc[index, 'lat']
    lon = data.node.loc[index, 'lon']
    node_idx = data.node.loc[index, 'index']
    node_id = data.node.loc[index, 'id']
    node_zone = data.node.loc[index, 'zone']
    area = data.node.loc[index, 'area']
    area_price = avg_national_prices.get(area, 'N/A')

    EUR_TO_ORE_PER_KWH = eur_to_nok / 1000 * 100
    price_nok = price * EUR_TO_ORE_PER_KWH
    area_price_nok = area_price * EUR_TO_ORE_PER_KWH if isinstance(area_price, (int, float)) else 'N/A'

    popup = folium.Popup(
        f"<b>Node index:</b> {node_idx}<br>"
        f"<b>Node id:</b> {node_id}<br>"
        f"<b>Zone:</b> {node_zone}<br>"
        f"<b>Price:</b> {price_nok:.2f} øre/kWh<br>"
        f"<b>National Price:</b> {f'{area_price_nok:.2f} øre/kWh' if isinstance(area_price_nok, (int, float)) else 'N/A'}<br>",
        max_width=200
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
            f"<b>Power Flow:</b> {flows[idx]:.2f} MW<br>"
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
        angle = math.degrees(math.atan2(nodeB['lon'] - nodeA['lon'], nodeB['lat'] - nodeA['lat']))

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
    AC_branch_path = grid_data_path / "branch.csv"
    DC_branch_path = grid_data_path / "dcbranch.csv"
    AC_branch_df = pd.read_csv(AC_branch_path)
    DC_branch_df = pd.read_csv(DC_branch_path)
    AC_cross_country_connections = AC_branch_df[AC_branch_df['node_from'].str[:2] != AC_branch_df['node_to'].str[:2]]
    DC_cross_country_connections = DC_branch_df[DC_branch_df['node_from'].str[:2] != DC_branch_df['node_to'].str[:2]]
    return AC_cross_country_connections, DC_cross_country_connections


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


"""FLOW DATA PLOTS"""
# Function to collect flow data
def collect_flow_data(db, time_max_min, cross_country_dict, interconnections_capacity, ac=True):
    flow_data = []
    branch_type = "AC" if ac else "DC"
    for branch_index, (node_from, node_to) in cross_country_dict.items():
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

def plot_duration_curve_by_year(row, DATE_START, OUTPUT_PATH_PLOTS, save_fig, duration_relative, tex_font):
    num_hours = len(row['load [MW]'])
    start_time = pd.to_datetime(DATE_START)
    timestamps = pd.date_range(start=start_time, periods=num_hours, freq='h')

    df_flow = pd.DataFrame({
        'timestamp': timestamps,
        'load [MW]': row['load [MW]']
    })
    df_flow['year'] = df_flow['timestamp'].dt.year

    for year, group in df_flow.groupby('year'):
        sorted_flows = sorted(group['load [MW]'], reverse=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        if duration_relative:
            percentile = [100 * (i + 1) / len(sorted_flows) for i in range(len(sorted_flows))]

            ax.plot(percentile, sorted_flows, label=f"From: {row['from']} To: {row['to']} in {year}", color='green')
            ax.axhline(y=row['capacity [MW]'], color='red', linestyle='--',
                       label=f"Max capacity: {row['capacity [MW]']:.2f} MW")
            ax.axhline(y=-row['capacity [MW]'], color='red', linestyle='--')
            ax.set_title(f"Duration Curve for {row['type']} Connection {row['from']} → {row['to']} in {year}")
            ax.set_xlabel("Percentile [%]")
            ax.set_ylabel("Load [MW]")
            ax = plt.gca()
            ax.xaxis.set_major_locator(plt.MultipleLocator(10))
            plt.setp(ax.xaxis.get_majorticklabels())
        else:
            # X-axis represents the number of hours
            hours = list(range(1, len(sorted_flows) + 1))

            plt.plot(hours, sorted_flows, label=f"From: {row['from']} To: {row['to']} in {year}", color='blue')
            plt.axhline(y=row['capacity [MW]'], color='red', linestyle='--',
                        label=f"Max capacity: {row['capacity [MW]']:.2f} MW")
            plt.axhline(y=-row['capacity [MW]'], color='red', linestyle='--')

            plt.xlabel("Number of Hours")
            plt.title(f"Sorted Load Curve for {row['type']} Connection {row['from']} → {row['to']} in {year}")
            plt.ylabel("Load [MW]")

            # Optionally, set x-axis limits or ticks if needed
            # For example, set major ticks every 1000 hours:
            ax = plt.gca()
            ax.xaxis.set_major_locator(plt.MultipleLocator(1000))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        ax.legend(loc='upper right')
        ax.grid(True)
        plt.tight_layout()
        if tex_font:
            plt.rcParams.update({
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Computer Modern Roman"]})
        if duration_relative:
            plot_file_name = OUTPUT_PATH_PLOTS / f"duration_curve_{row['from']}_{row['to']}_{year}.pdf"
        else:
            plot_file_name = OUTPUT_PATH_PLOTS / f"sorted_load_curve_{row['from']}_{row['to']}_{year}.pdf"
        if save_fig:
            plt.savefig(plot_file_name)
            print(f"Plot saved as: {plot_file_name}")
        plt.show()


def plot_by_year(row, DATE_START, OUTPUT_PATH_PLOTS, save_fig, interval, tex_font):

    num_hours = len(row['load [MW]'])
    start_time = pd.to_datetime(DATE_START)
    timestamps = pd.date_range(start=start_time, periods=num_hours, freq='h')

    df_flow = pd.DataFrame({
        'timestamp': timestamps,
        'load [MW]': row['load [MW]']
    })
    df_flow['year'] = df_flow['timestamp'].dt.year

    for year, group in df_flow.groupby('year'):
        plt.figure(figsize=(10, 6))
        plt.plot(group['timestamp'], group['load [MW]'], label=f"Load in {year}")
        plt.axhline(y=row['capacity [MW]'], color='red', linestyle='--',
                    label=f"Max capacity: {row['capacity [MW]']:.2f} MW")
        plt.axhline(y=-row['capacity [MW]'], color='red', linestyle='--')
        max_y = row['capacity [MW]']
        plt.ylim(-max_y - 100, max_y + 200)

        # Customize x-axis to show ticks every 'month_interval' months
        ax = plt.gca()  # Get current axes
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.title(f"Yearly Load flow for {row['type']} connection {row['from']} --> {row['to']} in {year}")
        plt.xlabel("Time")
        plt.ylabel("Load [MW]")
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.tight_layout()
        if tex_font:
            plt.rcParams.update({
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Computer Modern Roman"]})
        plot_file_name = OUTPUT_PATH_PLOTS / f"power_flow_{row['from']}_{row['to']}_{year}.pdf"
        if save_fig:
            plt.savefig(plot_file_name)
            print(f"Plot saved as: {plot_file_name}")
        plt.show()


def plot_duration_curve(row, OUTPUT_PATH_PLOTS, save_fig, duration_relative, tex_font):
    plt.figure(figsize=(10, 6))
    sorted_flows = sorted(row['load [MW]'], reverse=True)
    if duration_relative:
        # Calculate percentiles for the x-axis
        percentile = [100 * (i + 1) / len(sorted_flows) for i in range(len(sorted_flows))]

        plt.plot(percentile, sorted_flows, label=f"From: {row['from']} To: {row['to']}", color='green')
        plt.axhline(y=row['capacity [MW]'], color='red', linestyle='--',
                    label=f"Max capacity: {row['capacity [MW]']:.2f} MW")
        plt.axhline(y=-row['capacity [MW]'], color='red', linestyle='--')

        plt.xlabel("Percentile [%]")
        plt.title(f"Duration Curve for {row['type']} Connection {row['from']} → {row['to']}")

        ax = plt.gca()
        ax.xaxis.set_major_locator(plt.MultipleLocator(10))
        plt.setp(ax.xaxis.get_majorticklabels())

        plt.ylabel("Load [MW]")

    else:
        # X-axis represents the number of hours
        hours = list(range(1, len(sorted_flows) + 1))

        plt.plot(hours, sorted_flows, label=f"From: {row['from']} To: {row['to']}", color='blue')
        plt.axhline(y=row['capacity [MW]'], color='red', linestyle='--',
                    label=f"Max capacity: {row['capacity [MW]']:.2f} MW")
        plt.axhline(y=-row['capacity [MW]'], color='red', linestyle='--')

        plt.xlabel("Number of Hours")
        plt.title(f"Sorted Load Curve for {row['type']} Connection {row['from']} → {row['to']}")
        plt.ylabel("Load [MW]")

        # Optionally, set x-axis limits or ticks if needed
        # For example, set major ticks every 1000 hours:
        ax = plt.gca()
        ax.xaxis.set_major_locator(plt.MultipleLocator(1000))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    if tex_font:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"]})
    # Define the plot file name
    if duration_relative:
        plot_file_name = OUTPUT_PATH_PLOTS / f"duration_curve_{row['from']}_{row['to']}.pdf"
    else:
        plot_file_name = OUTPUT_PATH_PLOTS / f"sorted_load_curve_{row['from']}_{row['to']}.pdf"

    if save_fig:
        plt.savefig(plot_file_name)
        print(f"Plot saved as: {plot_file_name}")
    plt.show()


def plot_time_series(row, DATE_START, OUTPUT_PATH_PLOTS, save_fig, interval, tex_font):
    # Convert DATE_START to a pandas Timestamp if it's a string
    if isinstance(DATE_START, str):
        start_time = pd.to_datetime(DATE_START)
    elif isinstance(DATE_START, pd.Timestamp):
        start_time = DATE_START
    else:
        raise ValueError("DATE_START must be a string or pandas.Timestamp object.")

    # Determine the number of hours in the load data
    num_hours = len(row['load [MW]'])
    # Generate a range of timestamps at hourly intervals
    timestamps = pd.date_range(start=start_time, periods=num_hours, freq='h')
    # Create a pandas Series for load data with timestamps as the index
    load_series = pd.Series(data=row['load [MW]'], index=timestamps)

    plt.figure(figsize=(10, 6))
    plt.plot(load_series, label=f"From: {row['from']} To: {row['to']}")
    plt.axhline(y=row['capacity [MW]'], color='red', linestyle='--',
                label=f"Max capacity: {row['capacity [MW]']:.2f} MW")
    plt.axhline(y=-row['capacity [MW]'], color='red', linestyle='--')
    max_y = row['capacity [MW]']
    plt.ylim(-max_y - 100, max_y + 200)

    # Customize x-axis to show ticks every 'month_interval' months
    ax = plt.gca()  # Get current axes
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.title(f"Load flow for {row['type']} connection {row['from']} --> {row['to']}")
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
    plot_file_name = OUTPUT_PATH_PLOTS / f"power_flow_{row['from']}_{row['to']}.pdf"
    if save_fig:
        plt.savefig(plot_file_name)
        print(f"Plot saved as: {plot_file_name}")
    plt.show()


def plot_imp_exp_cross_border_Flow_NEW(db, DATE_START, time_max_min, grid_data_path, OUTPUT_PATH_PLOTS, by_year, duration_curve,
                                       duration_relative, save_fig, interval, check, tex_font):
    """
    Generates plots for cross-border AC and DC power flows.

    Parameters:
    - db: Database object to retrieve flow data.
    - grid_data_path: Path to the grid data.
    - time_max_min: Time range for the analysis.
    - OUTPUT_PATH_PLOTS: Directory path to save the plots.
    - by_year (bool): If True, generate separate plots for each year.
    - plot_duration_curve (bool): If True, plot duration curves instead of time series.
    - save_fig (bool): If True, save the plots as PDF files.
    """

    AC_interconnections, DC_interconnections = filter_cross_country_connections(grid_data_path)
    AC_interconnections_capacity = AC_interconnections['capacity']
    DC_interconnections_capacity = DC_interconnections['capacity']

    # Kan vurdere å legge inn zones slik at de kommer med i plottet... Krever litt koding

    # Get cross-country interconnections
    AC_cross_country_dict, DC_cross_country_dict = get_interconnections(grid_data_path)

    # Collect AC and DC flow data
    flow_data_AC = collect_flow_data(db, time_max_min, AC_cross_country_dict, AC_interconnections_capacity, ac=True)
    flow_data_DC = collect_flow_data(db, time_max_min, DC_cross_country_dict, DC_interconnections_capacity, ac=False)

    # Combine data into a single DataFrame
    flow_df = pd.concat([
        pd.DataFrame(flow_data_AC),
        pd.DataFrame(flow_data_DC)
    ], ignore_index=True)

    # Ensure OUTPUT_PATH_PLOTS is a Path object
    OUTPUT_PATH_PLOTS = pathlib.Path(OUTPUT_PATH_PLOTS)
    OUTPUT_PATH_PLOTS.mkdir(parents=True, exist_ok=True)

    if check:
        return flow_df
    # Plot import/ export load flow with respect to time for each interconnection
    for index, row in flow_df.iterrows():

        if by_year and duration_curve:
            # Plot duration curves for each year
            plot_duration_curve_by_year(row, DATE_START, OUTPUT_PATH_PLOTS, save_fig, duration_relative, tex_font)
        elif by_year and not duration_curve:
            plot_by_year(row, DATE_START, OUTPUT_PATH_PLOTS, save_fig, interval, tex_font)
        elif duration_curve and not by_year:
            plot_duration_curve(row, OUTPUT_PATH_PLOTS, save_fig, duration_relative, tex_font)
        else:
            plot_time_series(row, DATE_START, OUTPUT_PATH_PLOTS, save_fig, interval, tex_font)






