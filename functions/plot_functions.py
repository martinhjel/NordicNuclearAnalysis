from functions.global_functions import *
import seaborn as sns





# === Energy Mix ===
def getEnergyMix(data, database, timeMaxMin=None, relative=False, showTitle=True, variable="energy"):
    """
    Get energy, generation capacity or spilled energy per area per type

    Parameters
    ----------
    timeMaxMin : list of two integers
        Time range, [min,max]
    relative : boolean
        Whether to plot absolute (false) or relative (true) values
    variable : string ("energy","capacity","spilled")
        Which variable to plot (default is energy production)
    """

    if variable == "energy":
        print("Getting energy output from all generators...")
        gen_output = database.getResultGeneratorPowerSum(timeMaxMin)
    elif variable == "capacity":
        gen_output = data.generator.pmax
    elif variable == "spilled":
        gen_output = database.getResultGeneratorSpilledSums(timeMaxMin)
    else:
        print("Variable not valid")
        return

    df = data.generator[["node", "type", "pmax"]].merge(
        data.node[["id", "area"]], how="left", left_on="node", right_on="id"
    )
    df["VALUE"] = gen_output
    dfplot = df[["area", "type", "VALUE"]].groupby(["area", "type"]).sum()["VALUE"].unstack()

    if relative:
        dfplot = dfplot.mul(1 / dfplot.sum(axis=1), axis=0)

    return dfplot



def plotEnergyMix(
        data, database, areas=None, timeMaxMin=None, relative=False, showTitle=True, variable="energy", gentypes=None
):
    """
    Plot energy, generation capacity or spilled energy as stacked bars

    Parameters
    ----------
    areas : list of strings
        Which areas to include, default=None means include all
    timeMaxMin : list of two integers
        Time range, [min,max]
    relative : boolean
        Whether to plot absolute (false) or relative (true) values
    variable : string ("energy","capacity","spilled")
        Which variable to plot (default is energy production)
    gentypes : list
        List of generator types to include. None gives all.
    """


    dfplot = getEnergyMix(data, database, timeMaxMin=timeMaxMin, relative=relative, variable=variable)

    titles = {"energy": "Energy mix", "capacity": "Capacity mix", "spilled": "Energy spilled"}
    color_map = {
        "biomass": "#8c564b",
        "fossil_gas": "#d62728",
        "fossil_other": "#333333",
        "hydro": "#1f77b4",
        "nuclear": "#9467bd",
        "ror": "#17becf",
        "solar": "#ffbb78",
        "wind_off": "#aec7e8",
        "wind_on": "#2ca02c"
    }

    title = ""
    if variable in titles:
        title = titles[variable]
    if areas is None:
        areas = dfplot.index
    if gentypes is None:
        gentypes = dfplot.columns

    colors = [color_map.get(gt, "#333333") for gt in gentypes]  # fallback color if not found
    dfplot.loc[areas, gentypes].plot(kind="bar", stacked=True, color=colors)
    plt.legend()
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.reverse()
    labels.reverse()
    plt.legend(handles, labels, loc=2, bbox_to_anchor=(1.05, 1), borderaxespad=0.0)

    if showTitle:
        plt.title(title)
    plt.tight_layout()
    plt.show()
    return dfplot



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


from matplotlib.colors import Normalize
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import subprocess

def plot_storage_filling_area(storfilling, DATE_START, DATE_END, areas, interval, title=None, OUTPUT_PATH_PLOTS=None,
                              relative=None, plot_by_year=None, save_plot=None, duration_curve=None, tex_font=None,
                              legend=True, fig_size=(10, 6), plot_type=None, START=None, END=None):
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
    fig, ax = plt.subplots(figsize=fig_size)
    cmap = plt.get_cmap('Blues')
    # Define linestyles to differentiate areas
    linestyles = ['-', '--', ':', '-.'][:len(areas)]  # Cycle through linestyles for areas
    # Check LaTeX availability if tex_font is True
    if tex_font:
        try:
            subprocess.run(["pdflatex", "--version"], check=True, capture_output=True, text=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            # tex_font = False
            raise RuntimeError("LaTeX is not installed or not found in the system path. Please install LaTeX to use tex_font=True.")

    if tex_font:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ['cmr10'],
            "axes.formatter.use_mathtext": True,  # Fix cmr10 warning
            "axes.unicode_minus": False  # Fix minus sign rendering
        })
        print("Configured Matplotlib to use LaTeX with cmr10 font.")
    else:
        plt.rcParams.update({
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"]
        })
        print("Using default serif font (DejaVu Serif).")



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

        # Add legend for areas
        if legend:
            ax.legend(loc='upper left')



    else:
        years = sorted(storfilling['year'].unique())
        norm = Normalize(vmin=min(years), vmax=max(years))

        for idx, area in enumerate(areas):
            if area not in storfilling.columns:
                raise ValueError(f"{area} not found in storfilling DataFrame columns")
            for year in years:
                group = storfilling[storfilling['year'] == year]
                color = cmap(norm(year))
                if duration_curve:
                    sorted_values = group[area].sort_values(ascending=False).reset_index(drop=True)
                    ax.plot(sorted_values, label=None, color=color, linestyle=linestyles[idx])
                else:
                    ax.plot(group.index.dayofyear, group[area], label=None, color=color, linestyle=linestyles[idx])

        # Configure axes for yearly or duration curve plotting
        if not duration_curve:
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            ax.set_xlim(0, 364)

        # Add colorbar for years
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = plt.colorbar(sm, ax=ax, label='Year')
        cbar.set_ticks(years)  # Show only the years in the dataset
        cbar.set_label('Year', fontsize=10)

        # Add legend for areas (based on linestyles)
        area_legend = [Line2D([0], [0], color='black', linestyle=ls, label=area)
                       for area, ls in zip(areas, linestyles)]
        if legend:
            ax.legend(handles=area_legend, loc='upper left', title='Areas')

    # Configure axes
    ax.set_xlabel('Date' if not duration_curve else 'Hours')
    if tex_font:
        ax.set_ylabel('Storage Filling (\%)' if relative else 'Storage Filling (MWh)')
    else:
        ax.set_ylabel('Storage Filling (%)' if relative else 'Storage Filling (MWh)')
    if relative:
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        if tex_font:
            ax.set_yticklabels(['0\%', '20\%', '40\%', '60\%', '80\%', '100\%'])
        else:
            ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
        ax.set_ylim(0, 100)

    if title is not None:
        plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    # plt.subplots_adjust(bottom=0.3)  # Reserve space for legend


    # Show and/or save the plot
    if save_plot:
        if plot_type is None:
            plot_path = f'reservoir_filling_{START["year"]}-{END["year"]}_{"_".join(areas)}.pdf'
        else:
            plot_path = f'reservoir_filling_{START["year"]}-{END["year"]}_{"_".join(areas)}_type{plot_type}.svg'
        plt.savefig(OUTPUT_PATH_PLOTS / plot_path,  dpi=300, bbox_inches='tight')
        return plot_path
    plt.show()




#### FUNCTIONS TO PLOT NODAL PRICES

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
            "font.serif": ['cmr10'],
            "axes.formatter.use_mathtext": True,  # Fix cmr10 warning
            "axes.unicode_minus": False  # Fix minus sign rendering
        })
        print("Configured Matplotlib to use LaTeX with cmr10 font.")
    else:
        plt.rcParams.update({
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"]
        })
        print("Using default serif font (DejaVu Serif).")
    if save_plot_nodal:
        plt.savefig(OUTPUT_PATH_PLOTS / f'nodal_price_{zone}.pdf')
    plt.show()



def plot_nodal_prices_FromDB(data: GridData, node_prices, nodes_in_zone, zone, DATE_START, DATE_END, interval, TITLE, plot_all_nodes,
                      save_plot_nodal, OUTPUT_PATH_PLOTS, plot_by_year_nodal, duration_curve_nodal, tex_font):
    if plot_all_nodes:
        # Plotting the converted nodal prices for the NO2 nodes
        fig, ax1 = plt.subplots(figsize=(10, 6))
        if not duration_curve_nodal:
            for node in nodes_in_zone:
                ax1.plot(node_prices.index, node_prices[node],
                         label=f"Node {data.node.loc[node, 'id']}")

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
                ax1.plot(sorted_values, label=f"Node {data.node.loc[node, 'id']} (Duration Curve)")

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
            "font.serif": ['cmr10'],
            "axes.formatter.use_mathtext": True,  # Fix cmr10 warning
            "axes.unicode_minus": False  # Fix minus sign rendering
        })
        print("Configured Matplotlib to use LaTeX with cmr10 font.")
    else:
        plt.rcParams.update({
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"]
        })
        print("Using default serif font (DejaVu Serif).")
    if save_plot_nodal:
        plt.savefig(OUTPUT_PATH_PLOTS / f'nodal_price_{zone}.pdf')
    plt.show()



def plot_zonal_prices_FromDB(data: GridData, zone_prices, zones, DATE_START, DATE_END, interval, TITLE,
                      save_plot_nodal, OUTPUT_PATH_PLOTS, plot_by_year, duration_curve_nodal, tex_font):

    fig, ax1 = plt.subplots(figsize=(10, 6))
    if not plot_by_year:
        if not duration_curve_nodal:
            for zone in zones:

                ax1.plot(zone_prices.index, zone_prices[f'avg_price_{zone}'], label=f'Avg. Price - {zone}')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Price (EUR/MWh)')
            # Customize x-axis to show ticks every second month
            ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            ax1.set_xlim(pd.to_datetime(DATE_START), pd.to_datetime(DATE_END))
        else:
            for zone in zones:
                sorted_values = zone_prices[f'avg_price_{zone}'].sort_values(ascending=False).reset_index(drop=True)
                ax1.plot(sorted_values, label=f'Avg. Price - {zone}')
            ax1.set_xlabel('Rank')
            ax1.set_ylabel('Price (EUR/MWh)')
    else:
        for year in zone_prices['year'].unique():
            group = zone_prices[zone_prices['year'] == year]
            if duration_curve_nodal:
                for zone in zones:
                    sorted_values = group[f'avg_price_{zone}'].sort_values(ascending=False).reset_index(drop=True)
                    ax1.plot(sorted_values, label=f"{zone} - {year}")
                ax1.set_xlabel('Hour')
                ax1.set_ylabel('Price (EUR/MWh)')
            else:
                for zone in zones:
                    ax1.plot(group.index.dayofyear, group[f'avg_price_{zone}'], label=f"{zone} - {year}")
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
            "font.serif": ['cmr10'],
            "axes.formatter.use_mathtext": True,  # Fix cmr10 warning
            "axes.unicode_minus": False  # Fix minus sign rendering
        })
        print("Configured Matplotlib to use LaTeX with cmr10 font.")
    else:
        plt.rcParams.update({
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"]
        })
        print("Using default serif font (DejaVu Serif).")
    if save_plot_nodal:
        plt.savefig(OUTPUT_PATH_PLOTS / f"nodal_price_{ ', '.join(zones)}.pdf")
    plt.show()



def plotZonePriceMatrix(price_matrix, plot_config):
    import matplotlib.transforms as transforms
    fig, ax = plt.subplots(figsize=plot_config['fig_size'])
    heatmap = sns.heatmap(
        price_matrix.astype(float),
        annot=True,
        fmt='.0f',
        cmap=plot_config['colormap'],
        linewidths=0.5,
        cbar=False,
        ax=ax
    )
    # Set the colorbar label
    # Manually add the colorbar closer to the heatmap
    cbar = fig.colorbar(heatmap.collections[0], ax=ax, location='right', pad=plot_config['cbar_xpos'])  # smaller pad = closer
    cbar.set_label(plot_config['cbar_label'])

    if plot_config['title'] is not None:
        ax.set_title(plot_config['title'])
    plt.xlabel("Year")
    plt.ylabel("Zone")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=plot_config['rotation_y'], ha=plot_config['ha_y'],
                       va=plot_config['va_y'])
    ax.tick_params(axis='y', pad=10)  # Add padding to move labels away from plot
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=plot_config['rotation_x'], ha=plot_config['ha_x'],
    #                   va=plot_config['va_x'])

    dx = plot_config['x_label_xShift']  # Shift right by 5 points (1 point = 1/72 inch)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment(plot_config['ha_x'])  # or 'left'/'center'
        label.set_verticalalignment(plot_config['va_x'])  # or 'bottom'/'center'
        label.set_rotation(plot_config['rotation_x'])  # example rotation
        x, y = label.get_position()
        label.set_position((x, y + plot_config['x_label_yShift']))  # y-shift
        offset = transforms.ScaledTranslation(dx, 0, ax.figure.dpi_scale_trans)
        label.set_transform(label.get_transform() + offset)
    # Adjust layout to allocate more space on the left
    # plt.subplots_adjust(left=0.2)  # Increase left margin
    plt.tight_layout()
    if plot_config['save_fig']:
        filename = f"ZonePriceMatrix_{plot_config['version']}_{plot_config['start']}_{plot_config['end']}.pdf"
        plt.savefig(plot_config['OUTPUT_PATH_PLOTS'] / filename, dpi=plot_config['dpi'],
                    bbox_inches=plot_config['bbox_inches'])
        return filename
    plt.show()





#### FUNCTIONS TO CALCULATE AND PLOT -- Hydro Production, Reservoir Level and Inflow

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


def calculate_Hydro_Res_Inflow_FromDB(data: GridData, db: Database, DATE_START, area_OP, genType, time_max_min, relative_storage, include_pump):
    """
    Calculate hydro reservoir inflow, production, and storage filling.

    Parameters:
        data (powergama.Scenario): The scenario data containing profiles and grid information.
        DATE_START (str): The start date of the simulation period in 'YYYY-MM-DD' format.
        area_OP (str): The operational area for which to calculate inflow.
        genType (str): The type of generator (e.g., 'hydro').
        time_max_min (list): List containing the start and end indices for the simulation timeframe.
        include_pump (bool): If True, include pump power in the calculations.

    Returns:
        pd.DataFrame: A DataFrame containing resampled data for reservoir filling, hydro production, and inflow.
    """
    genTypeIdx = data.getGeneratorsPerAreaAndType()[area_OP][genType]
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
    inflow = inflow.iloc[time_max_min[0]:time_max_min[1]]  # +1 to include the max index
    print(f"Total inflow: {sum(inflow) / 1e6:.2f} TWh")

    # Hydro production
    hydro_production_sum = pd.DataFrame(db.getResultGeneratorPower(genTypeIdx, time_max_min)).sum(axis=1)

    # Storage filling
    storage_filling = getStorageFillingInAreasFromDB(data, db, areas=area_OP, generator_type=genType, relative_storage=relative_storage, timeMaxMin=time_max_min)
    storage_filling_percentage = [value * 100 for value in storage_filling]

    if include_pump:
        # Pump Power, it seems all generators have zero pump output
        pumpOutput = []
        for gen in genTypeIdx:
            # if res.grid.generator["pump_cap"][gen] > 0:
            pumpOutput.append(db.getResultPumpPower(gen, time_max_min))

    # Create DataFrame
    df = pd.DataFrame({
        'Reservoir Filling': storage_filling_percentage,
        'Hydro Production': hydro_production_sum.reset_index(drop=True),
        'Inflow': inflow.reset_index(drop=True)
    })
    df.index = pd.date_range(DATE_START, periods=(time_max_min[-1] - time_max_min[0]), freq='h')  # Set index to date range
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

        plt.title(TITLE + f'for weather years in period {DATE_START.year}-{DATE_END.year}')
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
            "font.serif": ['cmr10'],
            "axes.formatter.use_mathtext": True,  # Fix cmr10 warning
            "axes.unicode_minus": False  # Fix minus sign rendering
        })
        print("Configured Matplotlib to use LaTeX with cmr10 font.")
    else:
        plt.rcParams.update({
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"]
        })
        print("Using default serif font (DejaVu Serif).")
    if save_plot:
        plt.savefig(OUTPUT_PATH_PLOTS / 'hydro_production.pdf')
    plt.show()



#### CALCULATE AND PLOT -- PRODUCTION, LOAD AND PRICE

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



def calc_PLP_FromDB(data: GridData, db: Database, area_OP, DATE_START, time_max_min):
    time_period = time_max_min[-1] - time_max_min[0]
    nodes_in_zone_2 = data.node[data.node['area'] == area_OP].index.tolist()
    # Get nodal prices for all nodes in the zone in one step and apply conversion factor
    node_prices_2 = pd.DataFrame({node: getNodalPricesFromDB(db, node, time_max_min) for node in nodes_in_zone_2})
    node_prices_2.index = pd.date_range(DATE_START, periods=time_period, freq='h')
    avg_node_prices_2 = node_prices_2.sum(axis=1) / len(nodes_in_zone_2)

    # Load demand
    load_demand = getDemandPerAreaFromDB(data, db, area='NO', timeMaxMin=time_max_min)

    genHydroIdx = data.getGeneratorsPerAreaAndType()[area_OP]['hydro']
    hydro_production_sum = pd.DataFrame(db.getResultGeneratorPower(genHydroIdx, time_max_min)).sum(axis=1)
    hydro_production_sum.index = pd.date_range(DATE_START, periods=time_period, freq='h')

    df_plp = pd.DataFrame({
        'Production': hydro_production_sum,
        'Load': load_demand['sum'],
        'Price': avg_node_prices_2
    })

    df_plp.index = pd.date_range(DATE_START, periods=time_period, freq='h')

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
            "font.serif": ['cmr10'],
            "axes.formatter.use_mathtext": True,  # Fix cmr10 warning
            "axes.unicode_minus": False  # Fix minus sign rendering
        })
        print("Configured Matplotlib to use LaTeX with cmr10 font.")
    else:
        plt.rcParams.update({
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"]
        })
        print("Using default serif font (DejaVu Serif).")
    if save_fig:
        plt.savefig(OUTPUT_PATH_PLOTS / 'nodal_price_demand_hydro_production.pdf')
    plt.show()



##### PLOT LOAD SHEDDING
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
                "font.serif": ['cmr10'],
                "axes.formatter.use_mathtext": True,  # Fix cmr10 warning
                "axes.unicode_minus": False  # Fix minus sign rendering
            })
            print("Configured Matplotlib to use LaTeX with cmr10 font.")
        else:
            plt.rcParams.update({
                "text.usetex": False,
                "font.family": "serif",
                "font.serif": ["DejaVu Serif"]
            })
            print("Using default serif font (DejaVu Serif).")
        plt.show()




#### COLLECT AND PLOT PRODUCTION BY TYPE

def get_production_by_type(res, area_OP, time_max_min, DATE_START):

    generation_types = ['hydro', 'ror', 'nuclear', 'wind_on', 'wind_off', 'solar', 'fossil_gas', 'fossil_other',
                        'biomass']

    # Dictionary to store production data
    generation_data = {}

    # Iterate through generation types and fetch data
    for gen_type in generation_types:
        try:
            gen_idx = res.grid.getGeneratorsPerAreaAndType()[area_OP].get(gen_type, None)
            if gen_idx:
                production = pd.DataFrame(res.db.getResultGeneratorPower(gen_idx, time_max_min)).sum(axis=1)
                if production.sum() > 0:  # Ensure we only include nonzero production
                    generation_data[f"{gen_type.capitalize()}"] = production
        except Exception as e:
            print(f"Warning: Could not fetch data for {gen_type} in {area_OP}. Error: {e}")

    # Get Load Demand
    load_demand = res.getDemandPerArea(area=area_OP)

    # Get Avg Price for Area
    nodes_in_area = res.grid.node[res.grid.node['area'] == area_OP].index.tolist()
    node_prices = pd.DataFrame({
        node: res.getNodalPrices(node=node) for node in nodes_in_area
    })
    node_prices.index = pd.date_range(DATE_START, periods=time_max_min[-1], freq='h')
    avg_area_prices = node_prices.sum(axis=1) / len(nodes_in_area)

    # Create DataFrame with dynamically collected generation data
    df_gen = pd.DataFrame(generation_data)
    df_gen['Load'] = load_demand['sum']
    df_gen.index = pd.date_range(DATE_START, periods=time_max_min[-1], freq='h')

    # Define resampling rules dynamically
    resampling_rules = {col: 'sum' for col in df_gen.columns}

    # Resample the data based on the defined rules
    df_gen_resampled = df_gen.resample('7D').agg(resampling_rules)

    # Create price DataFrame
    df_prices = pd.DataFrame({'Price': avg_area_prices})
    df_prices.index = pd.date_range(DATE_START, periods=time_max_min[-1], freq='h')
    df_prices_resampled = df_prices.resample('1D').agg({'Price': 'mean'})

    total_production = df_gen.sum().sum()

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
                       ncol=4, frameon=False)  # Adjust ncol to fit the number of legend items

        plt.title(TITLE)
        plt.grid(True)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)  # Make space for the legend under the graph
        if tex_font:
            plt.rcParams.update({
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ['cmr10'],
                "axes.formatter.use_mathtext": True,  # Fix cmr10 warning
                "axes.unicode_minus": False  # Fix minus sign rendering
            })
            print("Configured Matplotlib to use LaTeX with cmr10 font.")
        else:
            plt.rcParams.update({
                "text.usetex": False,
                "font.family": "serif",
                "font.serif": ["DejaVu Serif"]
            })
            print("Using default serif font (DejaVu Serif).")
        if save_fig:
            plt.savefig(OUTPUT_PATH_PLOTS / 'production_prices_full_timeline.pdf', dpi=300, bbox_inches="tight")
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
                    "font.serif": ['cmr10'],
                    "axes.formatter.use_mathtext": True,  # Fix cmr10 warning
                    "axes.unicode_minus": False  # Fix minus sign rendering
                })
                print("Configured Matplotlib to use LaTeX with cmr10 font.")
            else:
                plt.rcParams.update({
                    "text.usetex": False,
                    "font.family": "serif",
                    "font.serif": ["DejaVu Serif"]
                })
                print("Using default serif font (DejaVu Serif).")
            if save_fig:
                plt.savefig(
                    OUTPUT_PATH_PLOTS / f'production_prices_{year}{"_duration_curve" if plot_duration_curve else ""}.pdf')
            plt.show()



"""FLOW DATA PLOTS"""

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
                "font.serif": ['cmr10'],
                "axes.formatter.use_mathtext": True,  # Fix cmr10 warning
                "axes.unicode_minus": False  # Fix minus sign rendering
            })
            print("Configured Matplotlib to use LaTeX with cmr10 font.")
        else:
            plt.rcParams.update({
                "text.usetex": False,
                "font.family": "serif",
                "font.serif": ["DejaVu Serif"]
            })
            print("Using default serif font (DejaVu Serif).")
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
                "font.serif": ['cmr10'],
                "axes.formatter.use_mathtext": True,  # Fix cmr10 warning
                "axes.unicode_minus": False  # Fix minus sign rendering
            })
            print("Configured Matplotlib to use LaTeX with cmr10 font.")
        else:
            plt.rcParams.update({
                "text.usetex": False,
                "font.family": "serif",
                "font.serif": ["DejaVu Serif"]
            })
            print("Using default serif font (DejaVu Serif).")
        plot_file_name = OUTPUT_PATH_PLOTS / f"power_flow_{row['from']}_{row['to']}_{year}.pdf"
        if save_fig:
            plt.savefig(plot_file_name)
            print(f"Plot saved as: {plot_file_name}")
        plt.show()


def plot_duration_curve(row, OUTPUT_PATH_PLOTS, plot_config):
    plt.figure(figsize=plot_config['fig_size'])
    sorted_flows = sorted(row['load [MW]'], reverse=True)
    if plot_config['duration_relative']:
        # Calculate percentiles for the x-axis
        percentile = [100 * (i + 1) / len(sorted_flows) for i in range(len(sorted_flows))]
        plt.axhline(y=row['capacity [MW]'], color='red', linestyle='--',
                    label=f"Max capacity: {row['capacity [MW]']:.2f} MW")
        plt.axhline(y=-row['capacity [MW]'], color='red', linestyle='--')
        plt.plot(percentile, sorted_flows, label=f"From: {row['from'][:3]} To: {row['to'][:3]}", color='green')


        plt.xlabel("Percentile [\%]")
        if plot_config['title'] is not None:
            plt.title(f"Duration Curve for {row['type']} Connection {row['from']} → {row['to']}")

        ax = plt.gca()
        ax.xaxis.set_major_locator(plt.MultipleLocator(10))
        plt.setp(ax.xaxis.get_majorticklabels())

        plt.ylabel("Flow [MW]")

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
    if plot_config['tex_font']:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ['cmr10'],
            "axes.formatter.use_mathtext": True,  # Fix cmr10 warning
            "axes.unicode_minus": False  # Fix minus sign rendering
        })
        print("Configured Matplotlib to use LaTeX with cmr10 font.")
    else:
        plt.rcParams.update({
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"]
        })
        print("Using default serif font (DejaVu Serif).")
    # Define the plot file name
    if plot_config['duration_relative']:
        plot_file_name = OUTPUT_PATH_PLOTS / f"duration_curve_{row['from']}_{row['to']}.pdf"
    else:
        plot_file_name = OUTPUT_PATH_PLOTS / f"sorted_load_curve_{row['from']}_{row['to']}.pdf"

    if plot_config['save_fig']:
        plt.savefig(plot_file_name, dpi=plot_config['dpi'], bbox_inches=plot_config['bbox_inches'])
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
            "font.serif": ['cmr10'],
            "axes.formatter.use_mathtext": True,  # Fix cmr10 warning
            "axes.unicode_minus": False  # Fix minus sign rendering
        })
        print("Configured Matplotlib to use LaTeX with cmr10 font.")
    else:
        plt.rcParams.update({
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"]
        })
        print("Using default serif font (DejaVu Serif).")
    plot_file_name = OUTPUT_PATH_PLOTS / f"power_flow_{row['from']}_{row['to']}.pdf"
    if save_fig:
        plt.savefig(plot_file_name)
        print(f"Plot saved as: {plot_file_name}")
    plt.show()


def plot_imp_exp_cross_border_Flow_NEW(db, DATE_START, time_max_min, grid_data_path, OUTPUT_PATH_PLOTS, by_year, duration_curve,
                                       duration_relative, save_fig, interval, check, tex_font, chosen_connections=None):
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

    if chosen_connections is not None:
        # Filter flow_data_AC
        flow_data_AC = [
            row for row in flow_data_AC
            if [row['from'], row['to']] in chosen_connections
        ]

        flow_data_DC = [
            row for row in flow_data_DC
            if [row['from'], row['to']] in chosen_connections
        ]

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





def plot_inflow_deviation(data):
    zones = ['NO1', 'NO2', 'NO3', 'NO4', 'NO5']
    # Initialize an empty dictionary to store inflow data
    inflow_data = {}

    # Extract inflow values and time index for each zone
    for zone in zones:
        inflow_data[zone] = np.array(data.profiles[f'inflow_{zone}'])
    # Create a DataFrame from the inflow data
    inflow_df = pd.DataFrame(inflow_data, index=data.profiles['time'])
    # Ensure the time index is a datetime object
    inflow_df.index = pd.to_datetime(inflow_df.index)
    # Add a 'year' column by extracting the year from the time index
    inflow_df['year'] = inflow_df.index.year
    # Group by year and calculate the average inflow for each zone
    average_inflow_per_year = inflow_df.groupby('year').mean()
    # Display the resulting DataFrame
    print(average_inflow_per_year)

    # Calculate percentage deviation from 1
    deviation = (average_inflow_per_year - 1) * 100

    # Create subplots
    num_zones = deviation.shape[1]
    fig, axes = plt.subplots(num_zones, 1, figsize=(10, 5 * num_zones), sharex=False)

    # Plot each zone in a separate subplot as a bar plot
    for i, zone in enumerate(deviation.columns):
        ax = axes[i] if num_zones > 1 else axes
        ax.bar(deviation.index, deviation[zone], label=f'{zone}')
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)  # Reference line at 0%
        ax.set_title(f'Deviation from Normal for Zone: {zone}', fontsize=12)
        ax.set_ylabel('Deviation (%)', fontsize=10)
        ax.set_xticks(deviation.index)  # Set x-ticks to the years
        ax.set_xticklabels(deviation.index, rotation=45, fontsize=8)  # Rotate and format year labels
        ax.grid()
        ax.legend()

    # Set common x-axis label
    plt.xlabel('Year', fontsize=12)
    plt.tight_layout()
    plt.show()

    return average_inflow_per_year


