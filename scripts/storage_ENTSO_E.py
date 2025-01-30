import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import pathlib


# There was only data spanning back to 2015 in the ENTSO-E database, easy to extract from Transparency Platform.
# The data was stored in a CSV file, which was uploaded to the data folder.

# Load the uploaded CSV file to analyze its content
file_path = pathlib.Path(f"../data//entso-e//Water Reservoirs and Hydro Storage Plants_201412290000-202101040000.csv")
data = pd.read_csv(file_path)


# Extract data for each year
weeks = data['Week'].str.extract(r'(\d+)').astype(int)[0]  # Extract only the week numbers
energy_2015 = data['Stored Energy Value 2015 [MWh] - Norway (NO)']
energy_2016 = data['Stored Energy Value 2016 [MWh] - Norway (NO)']
energy_2017 = data['Stored Energy Value 2017 [MWh] - Norway (NO)']
energy_2018 = data['Stored Energy Value 2018 [MWh] - Norway (NO)']
energy_2019 = data['Stored Energy Value 2019 [MWh] - Norway (NO)']
energy_2020 = data['Stored Energy Value 2020 [MWh] - Norway (NO)']


# Combine data into one DataFrame with a continuous timeline
all_years = pd.concat([
    pd.DataFrame({'Week': weeks, 'Year': 2015, 'Energy': energy_2015}),
    pd.DataFrame({'Week': weeks + 52, 'Year': 2016, 'Energy': energy_2016}),
    pd.DataFrame({'Week': weeks + 104, 'Year': 2017, 'Energy': energy_2017}),
    pd.DataFrame({'Week': weeks + 156, 'Year': 2018, 'Energy': energy_2018}),
    pd.DataFrame({'Week': weeks + 208, 'Year': 2019, 'Energy': energy_2019}),
    pd.DataFrame({'Week': weeks + 260, 'Year': 2020, 'Energy': energy_2020})
], ignore_index=True)

# Calculate the date for each week
reference_year = 2015  # Use 2015 as the base year for weekly dates
all_years['Date'] = pd.to_datetime(f'{reference_year}', format='%Y') + pd.to_timedelta((all_years['Week'] - 1) * 7, unit='D')

# Plot stored energy values for each year
plt.figure(figsize=(10, 6))
ax = plt.gca()

ax.plot(all_years['Date'], all_years['Energy'], label='NO - Storage')
# Format y-axis to display in TWh (instead of MWh)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1e6:.1f}'))

# Apply x-axis formatting
interval = 3  # Set interval for months
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

ax.set_xlim(pd.to_datetime("2015-01-01"), pd.to_datetime("2020-12-31"))

lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='lower left')  # , bbox_to_anchor=(1, 0))


plt.title('Reservoir Filling in NO for period 2015-2020 (ENTSO-E)')
plt.xlabel('Date')
plt.ylabel('Stored Energy [TWh]')
plt.grid(True)
plt.tight_layout()
plt.savefig(pathlib.Path("../plots/entso_e_plots/stored_energy_NO.pdf"))
plt.show()
