import pandas as pd
import pathlib

case = '2035'
scenario = 'BL'
version = 'v35'

# Get the base directory
try:
    # For scripts
    BASE_DIR = pathlib.Path(__file__).parent.parent
except NameError:
    # For notebooks or interactive shells
    BASE_DIR = pathlib.Path().cwd()

DATA_PATH = BASE_DIR / f'CASE_{case}' / f'scenario_{scenario}' / 'data' / 'system' / f'generator_{scenario}_{version}.csv'

# Henter ut consumer data fra Benchmark data
generator_data = pd.read_csv(DATA_PATH)

generator_data_updated = generator_data.copy()

# %%

solar_scaling = {
    'DK' : 1.0,
    'FI' : 1.79176,
    'NO' : 1.0,
    'SE' : 1.28534,
}


# Create a mask for all solar generators
for area in solar_scaling.keys():
    mask_solar = (generator_data_updated['type'] == 'solar') & (generator_data_updated['node'].str.startswith(area))
    generator_data_updated.loc[mask_solar, 'pmax'] *= solar_scaling[area]


# %%
onshoreWind_scaling = {
    'DK' : 1.0,
    'FI' : 1.0,
    'NO' : 1.0,
    'SE' : 1.43535,
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
hydro_scaling = {
    'FI' : 1.0,
    'NO' : 1.074,
    'SE' : 1.0,
}


# Create a mask for all solar generators
for area in hydro_scaling.keys():
    mask_hydro = (generator_data_updated['type'] == 'hydro') & (generator_data_updated['node'].str.startswith(area))
    generator_data_updated.loc[mask_hydro, 'pmax'] *= hydro_scaling[area]
    # mask_ror = (generator_data_updated['type'] == 'ror') & (generator_data_updated['node'].str.startswith(area))
    # generator_data_updated.loc[mask_ror, 'pmax'] *= hydro_scaling[area]




# %%
# Lagrer ny consumer skalert til case mappen
SAVE_PATH = BASE_DIR / f'CASE_{case}' / f'scenario_{scenario}' / 'data' / 'system' / f'generator_{scenario}_{version}_SCALED.csv'
generator_data_updated.to_csv(SAVE_PATH, index=False)