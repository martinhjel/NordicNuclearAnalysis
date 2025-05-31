import geopandas as gpd
import pathlib

try:
    # For scripts
    BASE_DIR = pathlib.Path(__file__).parent
except NameError:
    BASE_DIR = pathlib.Path().cwd() / 'power-market-app' / 'mapMaking' / 'geo'


# %%

# 1) Read the original GeoJSON
gdf = gpd.read_file(BASE_DIR / "UK.geojson")

# 2) Create a copy and simplify the geometry.
#    Increase or decrease the 'tolerance' value depending on how much
#    you want to reduce resolution (in the same units as your data).
gdf_simplified = gdf.copy()
gdf_simplified["geometry"] = gdf_simplified["geometry"].simplify(
    tolerance=0.01,  # e.g. 0.01 degrees if data is lat/lon
    preserve_topology=True
)

# 3) Save the simplified result to a new GeoJSON file
gdf_simplified.to_file(BASE_DIR / "UK_lowres.geojson", driver="GeoJSON")
print("Done! Created 'UK_lowres.geojson' with simplified geometry.")
