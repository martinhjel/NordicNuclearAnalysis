from idlelib.colorizer import color_config

import requests
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pathlib

import pytz


zones = {
    "NO1": "10YNO-1--------2",
    "NO2": "10YNO-2--------T",
    "NO3": "10YNO-3--------J",
    "NO4": "10YNO-4--------9",
    "NO5": "10Y1001A1001A48H",
    "DK1": "10YDK-1--------W",
    "DK2": "10YDK-2--------M",
    "SE1": "10Y1001A1001A44P",
    "SE2": "10Y1001A1001A45N",
    "SE3": "10Y1001A1001A46L",
    "SE4": "10Y1001A1001A47J",
    "FI":  "10YFI-1--------U",
    "DE":  "10Y1001A1001A82H", # 10Y1001A1001A83F
    "NL":  "10YNL----------L",
    "GB":  "10YGB----------A"
}

def main():
    API_KEY = "e278781c-c721-4675-8109-13caf4994141"
    zone_key = "NO5"
    zone = zones[zone_key]
    start_date = datetime(2019,1,1, 0)# "202001012300"
    end_date = datetime(2019, 12, 31, 23) # "202012312300"
    data = {}

    current_date = start_date
    while current_date < end_date:
        date_from = current_date.strftime("%Y%m%d%H%M")
        next_month = current_date + timedelta(days=31)  # Approximately one month
        date_to = min(next_month, end_date).strftime("%Y%m%d%H%M")

        monthly_data = getPrice(API_KEY, zone, date_from, date_to)
        if monthly_data:
            data.update(monthly_data)

        # Move to the next month
        current_date = next_month

    plotPrice(data, start_date, end_date, zone_key)
    df = pd.DataFrame(data.items(), columns=["time", "price"])
    x = 1

def remove_namespace(tree):
    """Remove namespaces in the passed XML tree for easier tag searching."""
    for elem in tree.iter():
        # Remove the namespace if present
        if "}" in elem.tag:
            elem.tag = elem.tag.split("}", 1)[1]
    return tree

def getPrice(API_KEY, zone, date_from, date_to):
    base_URL = "https://web-api.tp.entsoe.eu/api"
    params = {
        "documentType": "A44",
        "processType": "A01",
        "in_Domain": zone,
        "out_Domain": zone,
        "periodStart": date_from,
        "periodEnd": date_to,
        "securityToken": API_KEY
    }
    response = requests.get(base_URL, params=params)
    response.raise_for_status()

    if response.status_code == 200:
        try:
            root = remove_namespace(ET.fromstring(response.text))
            series = {}

            for time_series in root.findall(".//TimeSeries"):
                for period in time_series.findall(".//Period"):
                    resolution = period.find(".//resolution").text

                    if resolution != "PT60M":
                        continue

                    response_start = period.find(".//timeInterval//start").text
                    start_time = (
                        datetime.strptime(response_start, "%Y-%m-%dT%H:%MZ")
                        .replace(tzinfo=pytz.UTC)
                        .astimezone()
                    )

                    response_end = period.find(".//timeInterval//end").text
                    end_time = (
                        datetime.strptime(response_end, "%Y-%m-%dT%H:%MZ")
                        .replace(tzinfo=pytz.UTC)
                        .astimezone()
                    )

                    for point in period.findall(".//Point"):
                        position = point.find(".//position").text
                        price = point.find(".//price.amount").text
                        hour = int(position) - 1
                        series[start_time + timedelta(hours=hour)] = float(price)

                    current_time = start_time
                    last_price = series[current_time]

                    while current_time < end_time:
                        if current_time in series:
                            last_price = series[current_time]
                        else:
                            series[current_time] = last_price
                        current_time += timedelta(hours=1)

            return dict(sorted(series.items()))
        except Exception as e:
            print(f"Failed to parse data: {response.content}")
            raise e
    else:
        print(f"Failed to retrieve data: {response.status_code}")
        return None

def plotPrice(data, start_date, end_date, zone):
    # Convert the data into a DataFrame
    df = pd.DataFrame(list(data.items()), columns=["time", "price"])

    # Convert the "time" column to datetime if not already in datetime format
    df["time"] = pd.to_datetime(df["time"], utc=True)

    # Set "time" as the index for plotting
    df.set_index("time", inplace=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the data
    ax.plot(df.index, df['price'])
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (EUR/MWh)")

    # Rotate x-axis labels for better readability
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    ax.set_xlim(start_date, end_date)


    # Additional plot settings
    plt.title(f"Price in {zone}")
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig(pathlib.Path('../plots/base_case/price_NO2.pdf'))
    plt.show()






if __name__ == '__main__':
    main()