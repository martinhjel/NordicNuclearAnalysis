import geopandas as gpd
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import json
import plotly.graph_objects as go
import matplotlib.colors as mcolors

try:
    # For scripts
    BASE_DIR = pathlib.Path(__file__).parent
except NameError:
    BASE_DIR = pathlib.Path().cwd() / 'power-market-app' / 'mapMaking'


# gdf_UK = gpd.read_file(BASE_DIR / "geo/UK_lowres.geojson")
# gdf_DE = gpd.read_file(BASE_DIR / "geo/DE_LU.geojson")
# gdf_NL = gpd.read_file(BASE_DIR / "geo/NL.geojson")
# gdf_PL = gpd.read_file(BASE_DIR / "geo/PL.geojson")
# gdf_FI = gpd.read_file(BASE_DIR / "geo/FI.geojson")
# gdf_EE = gpd.read_file(BASE_DIR / "geo/EE.geojson")
# gdf_LT = gpd.read_file(BASE_DIR / "geo/LT.geojson")
# gdf_DK1 = gpd.read_file(BASE_DIR / "geo/DK_1.geojson")
# gdf_DK2 = gpd.read_file(BASE_DIR / "geo/DK_2.geojson")
# gdf_SE1 = gpd.read_file(BASE_DIR / "geo/SE_1.geojson")
# gdf_SE2 = gpd.read_file(BASE_DIR / "geo/SE_2.geojson")
# gdf_SE3 = gpd.read_file(BASE_DIR / "geo/SE_3.geojson")
# gdf_SE4 = gpd.read_file(BASE_DIR / "geo/SE_4.geojson")
# gdf_NO1 = gpd.read_file(BASE_DIR / "geo/NO_1.geojson")
# gdf_NO2 = gpd.read_file(BASE_DIR / "geo/NO_2.geojson")
# gdf_NO3 = gpd.read_file(BASE_DIR / "geo/NO_3.geojson")
# gdf_NO4 = gpd.read_file(BASE_DIR / "geo/NO_4.geojson")
# gdf_NO5 = gpd.read_file(BASE_DIR / "geo/NO_5.geojson")



# Read geojsons
gdf_list = [
    gpd.read_file(BASE_DIR / f"geo/{name}.geojson")
    for name in ["UK_lowres", "DE_LU", "NL", "PL", "FI", "EE", "LT",
                 "DK_1", "DK_2", "SE_1", "SE_2", "SE_3", "SE_4",
                 "NO_1", "NO_2", "NO_3", "NO_4", "NO_5"]
]

# gdf_zones = gpd.GeoDataFrame(pd.concat([gdf_UK, gdf_DE, gdf_NL, gdf_PL, gdf_FI, gdf_EE, gdf_LT,
#                                             gdf_DK1, gdf_DK2, gdf_SE1, gdf_SE2, gdf_SE3, gdf_SE4,
#                                             gdf_NO1, gdf_NO2, gdf_NO3, gdf_NO4, gdf_NO5], ignore_index=True),
#                                 geometry="geometry")

# Combine into single GeoDataFrame
gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True), geometry="geometry")


# gdf_zones = gdf_zones.drop(columns=["CTRY24CD", "CTRY24NM", "CTRY24NMW", "FID", "GlobalID", "BNG_E", "BNG_N",
#                                           "LONG", "LAT"])

# Drop unnecessary columns
gdf = gdf.drop(columns=[col for col in gdf.columns if col not in ["zoneName", "geometry"]])


# Add price data NVE øre/kWh
# price_map = {
#     "NO_1": 48.64, "NO_2": 58.16, "NO_3": 32.55, "NO_4": 27, "NO_5": 47.46,
#     "SE_1": 28.94, "SE_2": 28.46, "SE_3": 41.42, "SE_4": 57.65,
#     "DK_1": 82.23, "DK_2": 82.47, "FI": 52.75, "PL": 112.03,
#     "DE_LU": 92.64, "NL": 89.85, "EE": 99.74, "UK_England": 99.53,
#     "UK_Scotland": 99.53, "UK_Wales": 99.53, "UK_NorthernIreland": 99.53, "LT": 99.83,
# }

# Add price data NVE # (prices in EUR/MWh, converted from NVE's øre/kWh)
# price_map = {
#     "NO_1": 42.222, "NO_2": 50.486, "NO_3": 28.255, "NO_4": 23.437, "NO_5": 41.198,
#     "SE_1": 25.122, "SE_2": 24.705, "SE_3": 36.04, "SE_4": 50.043,
#     "DK_1": 71.38, "DK_2": 71.58, "FI": 45.789, "PL": 97.248,
#     "DE_LU": 80.416, "NL": 77.995, "EE": 86.579, "UK_England": 86.397,
#     "UK_Scotland": 86.397, "UK_Wales": 86.397, "UK_NorthernIreland": 86.397, "LT": 86.658,
# }

# Add price data BM EUR/MWh
# price_map = {
#     "NO_1": 53, "NO_2": 63, "NO_3": 43, "NO_4": 9, "NO_5": 46,
#     "SE_1": 22, "SE_2": 34, "SE_3": 61, "SE_4": 75,
#     "DK_1": 94, "DK_2": 83, "FI": 54, "PL": 107,
#     "DE_LU": 97, "NL": 86, "EE": 95, "UK_England": 99,
#     "UK_Scotland": 99, "UK_Wales": 99, "UK_NorthernIreland": 99, "LT": 84,
# }
# Add price data BM øre/kWh
price_map = {
    "NO_1": 61, "NO_2": 72, "NO_3": 49, "NO_4": 10, "NO_5": 53,
    "SE_1": 25, "SE_2": 39, "SE_3": 70, "SE_4": 86,
    "DK_1": 108, "DK_2": 95, "FI": 62, "PL": 123,
    "DE_LU": 112, "NL": 99, "EE": 109, "UK_England": 114,
    "UK_Scotland": 114, "UK_Wales": 114, "UK_NorthernIreland": 114, "LT": 96,
}

gdf["SpotPris"] = gdf["zoneName"].map(price_map)
gdf["centroid"] = gdf.geometry.centroid

# Manual label overrides (e.g., NO_4 shifted west)
label_offsets = {
    "NO_4": (-3.0, 0),  # move left
    "DK_2": (0, -0.5),   # move down
}

# Zones to hide labels for
hide_labels = {"UK_Scotland", "UK_Wales", "UK_NorthernIreland"}


# %%

plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ['cmr10'],
            "axes.formatter.use_mathtext": True,  # Fix cmr10 warning
            "axes.unicode_minus": False  # Fix minus sign rendering
        })

# Plot
fig, ax = plt.subplots(figsize=(5, 8))
# Example: Assume prices typically range from 0 to 150, and we want the midpoint (yellow) to be at 75
norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=75, vmax=150)

gdf.plot(column="SpotPris", cmap="RdYlGn_r", linewidth=0.8, edgecolor="0.8",
         legend=True, ax=ax, norm=norm,
         legend_kwds={
             "label": "Price (øre/kWh)",
             "orientation": "horizontal",
             "shrink": 0.5,  # Adjust the size of the color bar
             "pad": 0.05  # Adjust the distance between the map and the color bar
         })


# Add text labels
for _, row in gdf.iterrows():
    zone = row["zoneName"]
    price = row["SpotPris"]
    if pd.notnull(price) and zone not in hide_labels:
        x, y = row["centroid"].x, row["centroid"].y
        dx, dy = label_offsets.get(zone, (0, 0))  # default is no shift
        ax.text(x + dx, y + dy, f"{price:.0f}",
                ha='center', va='center', fontsize=12, fontweight='bold', color='black',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))


ax.axis("off")
# ax.set_title("Spot Prices by Price Zone", fontsize=16)

# Save as PDF or PNG
plt.tight_layout()
plt.savefig("price_map.pdf", dpi=300, bbox_inches='tight')
plt.show()


# %%
#
# import plotly.io as pio
#
#
# # 1. Project to a suitable CRS, then calculate accurate centroids
# gdf_zones_projected = gdf_zones.to_crs(epsg=3857)
# gdf_zones["centroid"] = gdf_zones_projected.geometry.centroid.to_crs(epsg=4326)
#
# # 2. Extract centroid coords into lat/lon columns
# gdf_zones["lat"] = gdf_zones["centroid"].apply(lambda p: p.y)
# gdf_zones["lon"] = gdf_zones["centroid"].apply(lambda p: p.x)
#
# # 3. Shift NO_4 slightly left (west) by subtracting from lon.
# #    Adjust the numeric value as needed so it looks good on the map.
# idx_no4 = gdf_zones["zoneName"] == "NO_4"
# gdf_zones.loc[idx_no4, "lon"] = gdf_zones.loc[idx_no4, "lon"] - 3
#
# # 4. Drop any non-GeoJSON-friendly columns before converting
# gdf_zones_for_geojson = gdf_zones.drop(columns=["centroid", "lat", "lon"])
# zones_geojson = json.loads(gdf_zones_for_geojson.to_json())
#
#
# # 5. Basic choropleth layer for colored polygons
# fig = px.choropleth_mapbox(
#     data_frame=gdf_zones,
#     geojson=zones_geojson,
#     locations="zoneName",
#     featureidkey="properties.zoneName",
#     color="SpotPris",
#     color_continuous_scale=["green", "yellow", "red"],
#     range_color=[0, 150],  # 👈 set min and max here
#     center={"lat": 61.6, "lon": 3},
#     zoom=3.6,
#     mapbox_style="carto-positron",
# )
# fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
#
# ignore_zones = ["UK_Scotland", "UK_Wales", "UK_NorthernIreland"]
# mask = ~gdf_zones["zoneName"].isin(ignore_zones)
#
# # 6. Create the "background box" markers
# box_size = 35
# bg_trace = go.Scattermapbox(
#     lat=gdf_zones.loc[mask, "lat"],
#     lon=gdf_zones.loc[mask, "lon"],
#     mode="markers",  # These markers will serve as the background box
#     marker=go.scattermapbox.Marker(
#         size=box_size,
#         color="white",  # Set the background color of your box here
#         opacity=0.8,
#     ),
#     hoverinfo="skip",  # Skip hover info for this trace
#     showlegend=False
# )
#
# # 7. Overlay the text labels
# text_trace = go.Scattermapbox(
#     lat=gdf_zones.loc[mask, "lat"],
#     lon=gdf_zones.loc[mask, "lon"],
#     mode="text",  # Display text only
#     text=[f"{price}" for price in gdf_zones.loc[mask, "SpotPris"]],
#     textposition="middle center",
#     textfont=dict(size=12, color="black"),
#     hoverinfo="skip",
#     showlegend=False
# )
#
# pio.renderers.default = "browser"  # Do not use "pdf" here
#
# fig.add_trace(bg_trace)
# fig.add_trace(text_trace)
#
# fig.show()  # Visual preview in browser
#
#
# # Then export
# fig.write_image("price_map.png")  # This should work if Kaleido is OK, Kaieldo need python 3.11


# %%
# import folium
# import branca.colormap as cm
# from folium.plugins import MiniMap, MeasureControl
# 
# # 2) Create folium map with dark background
# m = folium.Map(location=[60, 10], zoom_start=4, tiles="CartoDB dark_matter")
# 
# # 3) Build colormap
# min_price = gdf_zones["SpotPris"].min()
# max_price = gdf_zones["SpotPris"].max()
# cmap = cm.LinearColormap(["green", "yellow", "red"], vmin=min_price, vmax=max_price)
# cmap.caption = "Spot Price (EUR/MWh)"
# m.add_child(cmap)
# 
# # 4) Add polygons (GeoJson)
# def style_function(feature):
#     p = feature["properties"]["SpotPris"]
#     if p is None:
#         return {"fillColor": "gray", "color": "black", "weight": 1, "fillOpacity": 0.7}
#     return {"fillColor": cmap(p), "color": "black", "weight": 1, "fillOpacity": 0.7}
# 
# folium.GeoJson(
#     data=gdf_zones.to_json(),
#     style_function=style_function,
#     tooltip=folium.GeoJsonTooltip(fields=["zoneName", "SpotPris"]),
# ).add_to(m)
# 
# # 5) Suppose you have a flow DataFrame:
# df_flows = pd.DataFrame({
#     "from_zone": ["NO_1","NO_2","SE_3","SE_4"],
#     "to_zone":   ["SE_3","SE_4","NO_1","NO_2"],
#     "flow":      [1500, -300, 1200, 800]  # some MW, negative => direction reversed, etc
# })
# 
# # 6) Compute centroids for each zone
# zone_centers = {}
# for idx, row in gdf_zones.iterrows():
#     c = row.geometry.centroid
#     zone_centers[row["zoneName"]] = (c.y, c.x)  # lat, lon
# 
# flow_layer = folium.FeatureGroup(name="Zone Flows")
# 
# # 7) Add lines
# for idx, row in df_flows.iterrows():
#     fz = row["from_zone"]
#     tz = row["to_zone"]
#     flow = row["flow"]
# 
#     from_coords = zone_centers[fz]
#     to_coords = zone_centers[tz]
# 
#     # thickness
#     wt = max(1, min(abs(flow)/500, 8))  # tweak the scale
#     color = "green" if flow > 0 else "red"
# 
#     folium.PolyLine(
#         locations=[from_coords, to_coords],
#         color=color,
#         weight=wt,
#         opacity=0.8,
#         tooltip=f"{fz} → {tz}: {flow} MW"
#     ).add_to(flow_layer)
# 
# m.add_child(flow_layer)
# folium.LayerControl().add_to(m)  # So we can toggle flows on/off
# 
# 
# # Optional "box labels" for each zone
# labels_layer = folium.FeatureGroup(name="Zone Labels")
# for idx, row in gdf_zones.iterrows():
#     zone_name = row["zoneName"]
#     price = row["SpotPris"]
#     centroid = row.geometry.centroid
#     lat, lon = centroid.y, centroid.x
# 
#     label_html = f"""
#     <div style="font-size:12px; 
#                 background-color: white; 
#                 border: 1px solid black;
#                 border-radius: 4px;
#                 padding: 3px;">
#       <b>{zone_name}</b><br/>
#       Price: {price} EUR/MWh
#     </div>
#     """
#     folium.map.Marker(
#         [lat, lon],
#         icon=folium.features.DivIcon(html=label_html)
#     ).add_to(labels_layer)
# labels_layer.add_to(m)
# 
# # Add layer control
# folium.LayerControl().add_to(m)
# 
# # Add minimap, measure control, etc.
# minimap = MiniMap(toggle_display=True)
# m.add_child(minimap)
# 
# m.add_child(MeasureControl())
# 
# m.save("InteractiveMap.html")