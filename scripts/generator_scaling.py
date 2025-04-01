import pandas as pd
import pathlib

case = 'x_2035'
version = 'BL_v1'

# Get the base directory
try:
    # For scripts
    BASE_DIR = pathlib.Path(__file__).parent.parent
except NameError:
    # For notebooks or interactive shells
    BASE_DIR = pathlib.Path().cwd()

DATA_PATH = BASE_DIR / f'case_{case}' / 'data' / 'system' / f'generator_{version}_NOT_SCALED.csv'

# Henter ut consumer data fra Benchmark data
generator_data = pd.read_csv(DATA_PATH)

generator_data_updated = generator_data.copy()

# %%

solar_scaling = {
    'DK' : 19.43,
    'FI' : 282.0,
    'NO' : 72.0,
    'SE' : 1.51,
}


# Create a mask for all solar generators
for area in solar_scaling.keys():
    mask_solar = (generator_data_updated['type'] == 'solar') & (generator_data_updated['node'].str.startswith(area))
    generator_data_updated.loc[mask_solar, 'pmax'] *= solar_scaling[area]


# %%
onshoreWind_scaling = {
    'DK' : 1.4496,
    'FI' : 1.581,
    'NO' : 1.1,
    'SE' : 1.467,
}


# Create a mask for all onshore wind generators
for area in onshoreWind_scaling.keys():
    mask_wind_on = (generator_data_updated['type'] == 'wind_on') & (generator_data_updated['node'].str.startswith(area))
    generator_data_updated.loc[mask_wind_on, 'pmax'] *= onshoreWind_scaling[area]


# %%
offshoreWind_scaling = {
    'DK' : 5.32,
    'FI' : 113.64,
    'SE' : 12.66,
}


# Create a mask for all onshore wind generators
for area in offshoreWind_scaling.keys():
    mask_wind_off = (generator_data_updated['type'] == 'wind_off') & (generator_data_updated['node'].str.startswith(area))
    generator_data_updated.loc[mask_wind_off, 'pmax'] *= offshoreWind_scaling[area]




# %%

fossilOther_scaling = {
    'DK' : 0.8,
    'FI' : 0.8,
    'SE' : 0.8,
    'PL' : 0.8,
}


# %%
# Lagrer ny consumer skalert til case mappen
SAVE_PATH = BASE_DIR / f'case_{case}' / 'data' / 'system' / f'generator_{version}_Scaled.csv'
generator_data_updated.to_csv(SAVE_PATH, index=False)