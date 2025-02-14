import pandas as pd
import numpy as np
import pathlib

case = 'x'

# Get the base directory
try:
    # For scripts
    BASE_DIR = pathlib.Path(__file__).parent.parent
except NameError:
    # For notebooks or interactive shells
    BASE_DIR = pathlib.Path().cwd()

DATA_PATH = BASE_DIR / f'case_BM' / 'data' / 'system' / 'consumer.csv'

# Henter ut consumer data fra Benchmark data
consumer_data = pd.read_csv(DATA_PATH)

# %% Scale demand_avg for all nodes

scaling_factor = {'DK1' : 1,
                  'DK2' : 1,
                  'FI'  : 1,
                  'NO1' : 1,
                  'NO2' : 1,
                  'NO3' : 1,
                  'NO4' : 1,
                  'NO5' : 1,
                  'SE1' : 1,
                  'SE2' : 1,
                  'SE3' : 1,
                  'SE4' : 1,
                  'GB'  : 1,
                  'DE'  : 1,
                  'NL'  : 1,
                  'EE'  : 1,
                  'LT'  : 1,
                  'PL'  : 1,
                  }

# consumer_data['demand_avg'] = consumer_data['demand_avg'] * scaling_factor

# Scale by zone with different factor for each zone

zones = consumer_data['node'].str.split('_').str[0].unique()

# Update demand in node, for zone scaling factor
consumer_data_updated = consumer_data.copy()

for zone in zones:
    scaling_factor_zone = scaling_factor[zone]
    consumer_data_updated.loc[consumer_data_updated['node'].str.split('_').str[0] == zone, 'demand_avg'] = consumer_data.loc[consumer_data['node'].str.split('_').str[0] == zone, 'demand_avg'] * scaling_factor_zone

# Lagrer ny consumer skalert til case mappen
SAVE_PATH = BASE_DIR / f'case_{case}' / 'data' / 'system' / 'consumer_scaled.csv'
# consumer_data_updated.to_csv(SAVE_PATH)
