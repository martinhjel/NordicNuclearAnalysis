import requests
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pytz
import numpy as np
import os

# Soner og deres koder
zone_NO1 = "10YNO-1--------2"
zone_NO2 = "10YNO-2--------T"
zone_NO3 = "10YNO-3--------J"
zone_NO4 = "10YNO-4--------9"
zone_NO5 = "10Y1001A1001A48H"
zone_DK1 = "10YDK-1--------W"
zone_DK2 = "10YDK-2--------M"
zone_SE1 = "10Y1001A1001A44P"
zone_SE2 = "10Y1001A1001A45N"
zone_SE3 = "10Y1001A1001A46L"
zone_SE4 = "10Y1001A1001A47J"
zone_FI = "10YFI-1--------U"
# zone_DE = "10Y1001A1001A82H"    # Denne er feil...
# zone_NL = "10YNL----------L"
# zone_GB = "10YGB----------A"

zones = {
    "NO1": zone_NO1,
    "NO2": zone_NO2,
    "NO3": zone_NO3,
    "NO4": zone_NO4,
    "NO5": zone_NO5,
    "DK1": zone_DK1,
    "DK2": zone_DK2,
    "SE1": zone_SE1,
    "SE2": zone_SE2,
    "SE3": zone_SE3,
    "SE4": zone_SE4,
    "FI": zone_FI,
    # "NL": zone_NL,
    # "GB": zone_GB
}

def main(year):
    API_KEY = "e278781c-c721-4675-8109-13caf4994141"

    # Finner ikke Tyskland. Kontroller om det er noen connections som mangler. Har 15 her, i PowerGAMA har vi 19.
    connections = [
        # ("NO1", "SE3"),
        ("FI", "NO4"),
        # ("NO2", "DK1"),
        # ("NO2", "DE"),
        # ("NO2", "NL"),
        # ("NO2", "GB"),
        # ("NO3", "SE2"),
        # ("NO4", "SE2"),
        # ("NO4", "SE1"),
        #   ("SE1", "FI"),
        # ("SE1", "DE"),
        # ("SE3", "FI"),
        # ("SE4", "DE"),
        # ("DK2", "SE4"),
        # ("DK1", "SE3"),
        # ("DK1", "NL"),
        # ("DK2", "DE")
    ]

    start_date = datetime(year, 8, 1, 1)
    end_date = datetime(year, 11, 22, 23)

    for from_zone, to_zone in connections:
        in_zone_export = zones.get(from_zone)
        out_zone_export = zones.get(to_zone)

        if in_zone_export is None or out_zone_export is None:
            print(f"Tilkobling mellom {from_zone} og {to_zone} ikke funnet i 'zones'")
            continue

        # Hent eksport og import data
        export_data = get_combined_flow_data(API_KEY, in_zone_export, out_zone_export, start_date, end_date)
        import_data = get_combined_flow_data(API_KEY, out_zone_export, in_zone_export, start_date, end_date)

        # Lag DataFrames for eksport og import
        export_df = pd.DataFrame(export_data.items(), columns=["time", "flow (MW)"])
        export_df["export"] = export_df["flow (MW)"]
        export_df["import"] = 0

        import_df = pd.DataFrame(import_data.items(), columns=["time", "flow (MW)"])
        import_df["import"] = import_df["flow (MW)"]
        import_df["export"] = 0

        # Kombiner eksport- og importdata
        df = pd.concat([export_df, import_df]).groupby("time").sum().reset_index()

        # Lagre til CSV
        file_name = f"cross_border_{from_zone}_{to_zone}_{year}.csv"
        file_path = os.path.join(os.path.join('..', 'plots', 'base_case'), file_name)
        df.to_csv(file_path, index=False)
        print(f"Data for {from_zone} til {to_zone} i {year} lagret til {file_path}")

        # Plot data
        plotFlow(df, in_zone_export, out_zone_export, year)


def get_combined_flow_data(API_KEY, in_zone, out_zone, start_date, end_date):
    data = {}
    current_date = start_date

    while current_date < end_date:
        date_from = current_date.strftime("%Y%m%d%H%M")
        next_month = current_date + timedelta(days=31)
        date_to = min(next_month, end_date).strftime("%Y%m%d%H%M")

        flow_data = getCrossBorderFlow(API_KEY, in_zone, out_zone, date_from, date_to)
        if flow_data:
            data.update(flow_data)

        current_date = next_month

    return data


def getCrossBorderFlow(API_KEY, in_zone, out_zone, date_from, date_to):
    base_URL = "https://web-api.tp.entsoe.eu/api"
    params = {
        "documentType": "A11",
        "processType": "A16",
        "in_Domain": in_zone,
        "out_Domain": out_zone,
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
                        flow = point.find(".//quantity").text
                        hour = int(position) - 1
                        series[start_time + timedelta(hours=hour)] = float(flow)

                    current_time = start_time
                    last_flow = series[current_time]

                    while current_time < end_time:
                        if current_time in series:
                            last_flow = series[current_time]
                        else:
                            series[current_time] = last_flow
                        current_time += timedelta(hours=1)

            return dict(sorted(series.items()))
        except Exception as e:
            print(f"Feil ved parsing av data: {response.content}")
            raise e
    else:
        print(f"Feil ved henting av data: {response.status_code}")
        return None


def remove_namespace(tree):
    """Fjern namespaces i XML-treet for enklere søk."""
    for elem in tree.iter():
        if "}" in elem.tag:
            elem.tag = elem.tag.split("}", 1)[1]
    return tree


def plotFlow(df, in_zone, out_zone, year):
    in_zone_name = [key for key, value in zones.items() if value == in_zone][0]
    out_zone_name = [key for key, value in zones.items() if value == out_zone][0]

    df["import"] = df["import"].apply(lambda x: x * -1)
    df["export"] = df["export"].replace(0, np.nan)
    df["import"] = df["import"].replace(0, np.nan)

    df["flow"] = df["export"].combine_first(df["import"])
    df["flow"] = df["flow"].interpolate()

    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df["flow"], label="Flow (MW)", color="blue", linestyle='-')

    max_index = df["flow"].idxmax()
    min_index = df["flow"].idxmin()
    max_value = df["flow"].loc[max_index]
    min_value = df["flow"].loc[min_index]

    plt.scatter(max_index, max_value, color="red", label="Max Point", zorder=5)
    plt.scatter(min_index, min_value, color="red", label="Min Point", zorder=5)

    plt.annotate(f"Max: {max_value:.2f}",
                 (max_index, max_value),
                 textcoords="offset points",
                 xytext=(10, 10),
                 ha='center',
                 color="red")

    plt.annotate(f"Min: {min_value:.2f}",
                 (min_index, min_value),
                 textcoords="offset points",
                 xytext=(10, -15),
                 ha='center',
                 color="red")

    plt.title(f"Cross-border flow from {in_zone_name} to {out_zone_name} for {year}")
    plt.xlabel("Time (hours)")
    plt.ylabel("Flow (MW)")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main(2024)
