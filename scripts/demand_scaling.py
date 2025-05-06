import pandas as pd
import pathlib

CASE_YEAR = '2035'
SCENARIO = 'BL'
VERSION = 'v0'

# Get the base directory
try:
    # For scripts
    BASE_DIR = pathlib.Path(__file__).parent.parent
except NameError:
    # For notebooks or interactive shells
    BASE_DIR = pathlib.Path().cwd()

DATA_PATH = BASE_DIR / f'CASE_{CASE_YEAR}' / f'scenario_{SCENARIO}' / 'data' / 'system' / 'OLD' / f'consumer_{SCENARIO}_{VERSION}.csv'

# Henter ut consumer data fra Benchmark data
consumer_data = pd.read_csv(DATA_PATH)
# %%
profilePATH = BASE_DIR / f'CASE_{CASE_YEAR}' / f'scenario_{SCENARIO}' / 'data' / f'timeseries_profiles.csv'
profile = pd.read_csv(profilePATH)


# %%

### FOR 2035 ###


# Print max normaliserte load for each zone
print("SE1: ", profile['load_SE1'].max())
print("SE2: ", profile['load_SE2'].max())
print("SE3: ", profile['load_SE3'].max())
print("SE4: ", profile['load_SE4'].max())
print("NO1: ", profile['load_NO1'].max())
print("NO2: ", profile['load_NO2'].max())
print("NO3: ", profile['load_NO3'].max())
print("NO4: ", profile['load_NO4'].max())
print("NO5: ", profile['load_NO5'].max())
print("DK1: ", profile['load_DK1'].max())
print("DK2: ", profile['load_DK2'].max(), "DK2 MAX ID: ", profile['load_DK2'].idxmax())
print("FI: ", profile['load_FI'].max())

print(profile.head())

profiles_sorted = profile['load_DK2'].sort_values(ascending=False)
percentile_97_5_DK2 = profile['load_DK2'].quantile(0.9997)
percentile_97_5_SE2 = profile['load_SE2'].quantile(0.9997)

# %%
# Get all zones
zones = consumer_data['node'].str.split('_').str[0].unique()
demand_zones = {}
# Collect sum demand average before change for each zone
for zone in zones:
    print(consumer_data.loc[consumer_data['node'].str.split('_').str[0] == zone, 'demand_avg'])
    demand_zones[zone] = {'OldDemandAvg': sum(consumer_data.loc[consumer_data['node'].str.split('_').str[0] == zone, 'demand_avg'])}

# Max load before
for zone in zones:
    demand_zones[zone]['OldMax'] = demand_zones[zone]['OldDemandAvg'] * profile[f'load_{zone}'].quantile(0.9997)

# D/L
for zone in zones:
    demand_zones[zone]['Factor'] = demand_zones[zone]['OldDemandAvg'] / demand_zones[zone]['OldMax']

# New demand added to the max demand of each zone
newDemand = {'DK1' : 2300,
             'DK2' : 1300,
             'FI'  : 6750,
             'NO1' : 500,
             'NO2' : 3300,
             'NO3' : 900,
             'NO4' : 2300,
             'NO5' : 580,
             'SE1' : 8750,
             'SE2' : 3000,
             'SE3' : 3100,
             'SE4' : 0,
             'GB'  : 0,
             'DE'  : 0,
             'NL'  : 0,
             'EE'  : 0,
             'LT'  : 0,
             'PL'  : 0,
             }

# calculate new max load for each zone
for zone in zones:
    demand_zones[zone]['NewMax'] = demand_zones[zone]['OldMax'] + newDemand[zone]

# Calculate new demand average for each zone based on the new max load, using the factor calculated earlier
for zone in zones:
    demand_zones[zone]['NewDemandAvg'] = demand_zones[zone]['NewMax'] * demand_zones[zone]['Factor']

# %%

# Get percentage of demand in node within the zone

# Calculate demand percentage per node within its zone
demand_nodes = {}
for node in consumer_data['node'].unique():
    zone = node.split('_')[0]
    zone_demand = demand_zones[zone]['OldDemandAvg']
    node_demand = consumer_data.loc[consumer_data['node'] == node, 'demand_avg'].iloc[0]
    demand_nodes[node] = (node_demand / zone_demand)


# Calculate new demand for each node
# Update demand in node, for zone scaling factor
consumer_data_updated = consumer_data.copy()

for node in consumer_data_updated['node'].unique():
    zone = node.split('_')[0]
    demand_percentage = demand_nodes[node]
    new_demand_avg = demand_zones[zone]['NewDemandAvg'] * demand_percentage
    consumer_data_updated.loc[consumer_data_updated['node'] == node, 'demand_avg'] = new_demand_avg

# %%
# Lagrer ny consumer skalert til case mappen
SAVE_PATH = BASE_DIR / f'CASE_{CASE_YEAR}' / f'scenario_{SCENARIO}' / 'data' / 'system' / f'consumer_{SCENARIO}_{VERSION}_SCALED.csv'
consumer_data_updated.to_csv(SAVE_PATH, index=False)

# %% Scale demand_avg for all nodes


### For 2025 ###

scaling_factor = {'DK1' : 2658/2571,
                  'DK2' : 1670/1616,
                  'FI'  : 9182/9046,
                  'NO1' : 4054/3994,
                  'NO2' : 4169/4107,
                  'NO3' : 3284/3235,
                  'NO4' : 2330/2295,
                  'NO5' : 1974/1954,
                  'SE1' : 1200/1232,
                  'SE2' : 1679/1723,
                  'SE3' : 9302/9544,
                  'SE4' : 2446/2520,
                  'GB'  : 1.0,
                  'DE'  : 1.0,
                  'NL'  : 1.0,
                  'EE'  : 1.0,
                  'LT'  : 1.0,
                  'PL'  : 1.0,
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
SAVE_PATH = BASE_DIR / f'CASE_{CASE_YEAR}' / f'scenario_{SCENARIO}' / 'data' / 'system' / f'consumer_{SCENARIO}_{VERSION}_SCALED.csv'
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