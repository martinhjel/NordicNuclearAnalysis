import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates


countries  = ['norway', 'sweden', 'finland', 'denmark']

data = pd.DataFrame()
for cunt in countries :
    file_path = pathlib.Path().cwd() / f"files/entso-e/Total Load Forecast - Year Ahead_{cunt}.csv"

    # Read CSV file for the current country
    temp = pd.read_csv(file_path)

    # Add lower and upper data for the current country
    data[f"{cunt} lower"] = temp.iloc[:, 1]  # Second column
    data[f"{cunt} upper"] = temp.iloc[:, 2]  # Third column

print(data)

reference_year = 2024
data['Date'] = pd.to_datetime(f'{reference_year}', format='%Y') + pd.to_timedelta((data.index + 1) * 7, unit='D')


# Calculate the median for each country
for cunt in countries :
    data[f"{cunt} median"] = (data[f"{cunt} lower"] + data[f"{cunt} upper"]) / 2

# Create subplots for each country
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("Load Forecast for The Nordic Countries for 2024", fontsize=24)

for i, country in enumerate(countries):
    ax = axs[i // 2, i % 2]  # Determine subplot position

    # Calculate median if not already calculated
    if f"{country} median" not in data.columns:
        data[f"{country} median"] = (data[f"{country} lower"] + data[f"{country} upper"]) / 2

    # Plot upper and lower bounds with a filled area in between
    ax.fill_between(
        data["Date"],
        data[f"{country} lower"],
        data[f"{country} upper"],
        alpha=0.3,
        label=f"Load Range"
    )
    ax.plot(data["Date"], data[f"{country} median"], label=f"Load Median", linestyle="--", color="black")

    # Customize x-axis to show ticks every second month
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))  # Set interval to 1 month
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # Set x-axis limits to cover the entire data range
    ax.set_xlim(pd.to_datetime(f"{reference_year}-01-01"), pd.to_datetime(f"{reference_year}-12-31"))

    # Add labels, grid, and legend
    ax.set_title(f"{country.capitalize()} Load Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Load Profile [MW]")
    ax.legend(loc="lower left")
    ax.grid(True)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.95])

# plt.savefig(pathlib.Path("../plots/entso_e_plots/load_forecast_bounds.pdf"))
# Show the plot
plt.show()

# %%
import pathlib
import pandas as pd
import matplotlib.pyplot as plt

# Define the areas
areas = ['EE', 'LT', 'PL']
data = {}

# Desired aggregation interval
aggregation_interval = '7D'  # Change to '1D' for daily, '1H' for hourly, etc.

# Load and aggregate data
for area in areas:
    file_path = pathlib.Path(f"data/nordic/data_timeseries/load_{area}.csv")
    temp = pd.read_csv(file_path)

    # Convert the time column to datetime
    temp['Time'] = pd.to_datetime(temp.iloc[:, 1])  # Assume 2nd column is Time

    # Set the Time column as the index for resampling
    temp.set_index('Time', inplace=True)

    # Aggregate the data using mean
    aggregated = temp.iloc[:, 1].resample(aggregation_interval).mean()  # Assume 3rd column is Load

    # Store the aggregated data
    data[f"Time {area}"] = aggregated.index
    data[area] = aggregated.values

# Plot the aggregated load profile for each area
plt.figure(figsize=(12, 6))
for area in areas:
    plt.plot(data[f"Time {area}"], data[area], label=area)

# Customize the plot
plt.title(f"Aggregated Load Profile for EE, LT, and PL ({aggregation_interval} resolution)")
plt.xlabel("Time")
plt.ylabel("Load (MW)")
plt.legend()
plt.grid()
plt.tight_layout()

# Show the plot
plt.show()
