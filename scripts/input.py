import powergama
import pathlib
import folium
import pandas as pd
import os
from pygments.styles.dracula import DraculaStyle
# from networkx.classes import nodes
# from pandas.io.formats.style import color



def read_data(case, scenario, version):
    base_path = pathlib.Path(f"../CASE_{case}/scenario_{scenario}/data/system/")
    system_data = powergama.GridData()
    system_data.readGridData(nodes=base_path/f"node_{scenario}_{version}.csv",
                        ac_branches=base_path/f"branch_{scenario}_{version}.csv",
                        dc_branches=base_path/f"dcbranch_{scenario}_{version}.csv",
                        generators=base_path/f"generator_{scenario}_{version}.csv",
                        consumers=base_path/f"consumer_{scenario}_{version}.csv")

    return system_data

def node_plot(scenario_data):
    # Nodes, Generators, and Consumers
    node_df = scenario_data.node
    generator_df = scenario_data.generator
    consumer_df = scenario_data.consumer

    # Combine multiple generators per node into a summary
    if 'desc' in generator_df.columns and 'type' in generator_df.columns:
        generator_summary = (
            generator_df.groupby('node', group_keys=False)
            .apply(lambda x: '<br>'.join([
                f"{' '.join(row['desc'].split()[1:]) if pd.notna(row['desc']) and row['desc'].strip() != '' else row['type']}: {row['pmax']:.2f} MW"
                for _, row in x.iterrows()
            ]))
            .reset_index(name='generator_info')  # Use reset_index with name for compatibility
        )
    else:
        generator_summary = pd.DataFrame(columns=['node', 'generator_info'])

    # Combine multiple consumers per node into a summary
    if 'Load' in consumer_df.columns and 'demand_avg' in consumer_df.columns:
        consumer_summary = (
            consumer_df.groupby('node', group_keys=False)
            .apply(lambda x: '<br>'.join([
                f"{row['Load']}: {row['demand_avg']:.2f} MW"
                for _, row in x.iterrows()
            ]))
            .reset_index(name='consumer_info')  # Use reset_index with name for compatibility
        )
    else:
        consumer_summary = pd.DataFrame(columns=['node', 'consumer_info'])

    # Merge summaries with node data
    merged_df = node_df.merge(generator_summary, left_on='id', right_on='node', how='left')
    merged_df = merged_df.merge(consumer_summary, left_on='id', right_on='node', how='left')

    # Calculate map center
    lat_mean = node_df['lat'].mean()
    lon_mean = node_df['lon'].mean()
    map = folium.Map(location=[lat_mean, lon_mean], zoom_start=5)

    # Add nodes with generator and consumer info to the map
    for _, row in merged_df.iterrows():
        generator_info = row['generator_info'] if pd.notna(row['generator_info']) else "No installed capacity"
        consumer_info = row['consumer_info'] if pd.notna(row['consumer_info']) else "No average demand"

        # Popup text
        popup_text = f"""
        <b>Node ID:</b> {row['id']}<br>
        <b>Zone:</b> {row['zone']}<br><br>
        <b>Installed Capacity:</b><br>{generator_info}<br><br>
        <b>Average Demand:</b><br>{consumer_info}
        """
        # marker_color = "orange" if "SMR" in generator_info else "purple" if "data_centre" in row['id'] else "blue"
        marker_color = "orange" if "NEW" in generator_info else "purple" if "data_centre" in row['id'] else "blue"

        folium.Marker(
            location=[row['lat'], row['lon']],
            popup=folium.Popup(popup_text, max_width=400),
            icon=folium.Icon(color=marker_color, icon="")
        ).add_to(map)

    return map

def dc_link_plot(map, scenario_data):
    # DC links
    dc_branch_df = scenario_data.dcbranch
    node_df = scenario_data.node
    dc_link_df = dc_branch_df.merge(node_df[['id', 'lat', 'lon']], left_on='node_from', right_on='id', suffixes=('', '_from'))
    dc_link_df = dc_link_df.merge(node_df[['id', 'lat', 'lon']], left_on='node_to', right_on='id', suffixes=('', '_to'))

    for idx, row in dc_link_df.iterrows():
        popup_text = (
            f"<b>DC-link:</b> {row['node_from']} - {row['node_to']}<br>"
            f"<b>Capacity:</b> {row['capacity']:.3f} MW"
        )
        popup = folium.Popup(popup_text, max_width=300)
        folium.PolyLine(
            locations=[[row['lat'], row['lon']], [row['lat_to'], row['lon_to']]],
            color='blue',
            popup=popup
        ).add_to(map)

    return map

def ac_branch_plot(map, scenario_data):
    # AC links
    ac_branch_df = scenario_data.branch
    node_df = scenario_data.node
    ac_branch_df = ac_branch_df.merge(node_df[['id', 'lat', 'lon']], left_on='node_from', right_on='id', suffixes=('', '_from'))
    ac_branch_df = ac_branch_df.merge(node_df[['id', 'lat', 'lon']], left_on='node_to', right_on='id', suffixes=('', '_to'))

    for idx, row in ac_branch_df.iterrows():
        popup_text = (
            f"<b>Branch:</b> {row['node_from']} - {row['node_to']}<br>"
            f"<b>Capacity:</b> {row['capacity']:.3f} MW"
        )
        popup = folium.Popup(popup_text, max_width=300)
        folium.PolyLine(
            locations=[[row['lat'], row['lon']], [row['lat_to'], row['lon_to']]],
            color='red',
            popup=popup
        ).add_to(map)

    return map

def grid_plot(scenario_data):
    map = node_plot(scenario_data)                  # Generate map with nodes
    map = dc_link_plot(map, scenario_data)          # Add DC links to the map
    map = ac_branch_plot(map, scenario_data)        # Add AC links to the map
    return map

def main():
    case_year = 2035
    scenario = 'VDT'
    version = 'v9_sens'
    scenario_data = read_data(case_year, scenario, version)
    grid_map = grid_plot(scenario_data)

    output_dir = os.path.join("..", f"CASE_{case_year}", f"scenario_{scenario}", "results", "input")
    output_file = os.path.join(output_dir, f"grid_plot_case_{case_year}_{version}.html")
    grid_map.save(output_file)
    print(f"Grid plot saved to {output_file}")

if __name__ == "__main__":
    main()
