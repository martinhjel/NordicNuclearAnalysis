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

DATA_PATH = BASE_DIR / f'case_{case}' / 'data' / 'system' / f'consumer_{version}_NOT_SCALED.csv'

# Henter ut consumer data fra Benchmark data
consumer_data = pd.read_csv(DATA_PATH)

# %% Scale demand_avg for all nodes

scaling_factor = {'DK1' : 1.474,
                  'DK2' : 1.474,
                  'FI'  : 1.405,
                  'NO1' : 1.295,
                  'NO2' : 1.295,
                  'NO3' : 1.295,
                  'NO4' : 1.295,
                  'NO5' : 1.295,
                  'SE1' : 1.386,
                  'SE2' : 1.386,
                  'SE3' : 1.386,
                  'SE4' : 1.386,
                  'GB'  : 1.219,
                  'DE'  : 1.2998,
                  'NL'  : 1.459,
                  'EE'  : 1.88,
                  'LT'  : 1.636,
                  'PL'  : 1.455,
                  }

# scaling_factor_2025 = {'DK1' : 0.81928,
#                        'DK2' : 0.88433,
#                        'FI'  : 0.91354,
#                        'NO1' : 0.88382,
#                        'NO2' : 0.90503,
#                        'NO3' : 0.91575,
#                        'NO4' : 0.90285,
#                        'NO5' : 0.90684,
#                        'SE1' : 0.89649,
#                        'SE2' : 0.84376,
#                        'SE3' : 0.90334,
#                        'SE4' : 0.84376,
#                        'GB'  : 0.81682,
#                        'DE'  : 0.93011,
#                        'NL'  : 0.97119,
#                        'EE'  : 0.97055,
#                        'LT'  : 1.01244,
#                        'PL'  : 0.99785,
#                        }

# consumer_data['demand_avg'] = consumer_data['demand_avg'] * scaling_factor

# Scale by zone with different factor for each zone

zones = consumer_data['node'].str.split('_').str[0].unique()

# Update demand in node, for zone scaling factor
consumer_data_updated = consumer_data.copy()

for zone in zones:
    scaling_factor_zone = scaling_factor[zone]
    consumer_data_updated.loc[consumer_data_updated['node'].str.split('_').str[0] == zone, 'demand_avg'] = consumer_data.loc[consumer_data['node'].str.split('_').str[0] == zone, 'demand_avg'] * scaling_factor_zone


# Lagrer ny consumer skalert til case mappen
SAVE_PATH = BASE_DIR / f'case_{case}' / 'data' / 'system' / f'consumer_{version}_Scaled.csv'
consumer_data_updated.to_csv(SAVE_PATH, index=False)

# %%
# Add a new column 'zone' to both dataframes by extracting the part before the underscore
consumer_data['zone'] = consumer_data['node'].str.split('_').str[0]
consumer_data_updated['zone'] = consumer_data_updated['node'].str.split('_').str[0]

# Group by zone and calculate total demand for original and updated data
zone_old = consumer_data.groupby('zone')['demand_avg'].sum().reset_index().rename(columns={'demand_avg': 'demand_old'})
zone_new = consumer_data_updated.groupby('zone')['demand_avg'].sum().reset_index().rename(columns={'demand_avg': 'demand_new'})

# Merge the two summaries into one dataframe
df_combined = pd.merge(zone_old, zone_new, on='zone')

# Calculate the absolute change and the percentage change
df_combined['change'] = df_combined['demand_new'] - df_combined['demand_old']
df_combined['percentage_change'] = (df_combined['change'] / df_combined['demand_old']) * 100



# %%
import matplotlib.pyplot as plt
import numpy as np
# Set up the bar plot parameters
x = np.arange(len(df_combined['zone']))  # label locations
width = 0.35  # width of the bars

fig, ax = plt.subplots(figsize=(10, 6))

# Create the bars for old and new demand
rects1 = ax.bar(x - width/2, df_combined['demand_old'], width, label='Old Demand')
rects2 = ax.bar(x + width/2, df_combined['demand_new'], width, label='New Demand')

# Add labels, title, and custom x-axis tick labels
ax.set_xlabel('Zone')
ax.set_ylabel('Demand (MW)')
ax.set_title('Comparison of Old vs New Average Demand by Zone')
ax.set_xticks(x)
ax.set_xticklabels(df_combined['zone'])
ax.legend()

# Optionally add labels to each bar for clarity
def autolabel(rects):
    """Attach a text label above each bar displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    rotation=90)

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.show()