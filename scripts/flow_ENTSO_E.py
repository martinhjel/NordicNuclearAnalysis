import os
import time
from concurrent.futures import ThreadPoolExecutor
from entsoe import EntsoePandasClient
import pandas as pd
from requests.exceptions import ConnectionError

# Oppsett av API-klient
API_KEY = "e278781c-c721-4675-8109-13caf4994141"
client = EntsoePandasClient(api_key=API_KEY)

# Oppdaterte EIC-koder
price_areas = {
    "NO1": "10YNO-1--------2",
    "NO2": "10YNO-2--------T",
    "NO3": "10YNO-3--------J",
    "NO4": "10YNO-4--------9",
    "NO5": "10Y1001A1001A48H",
    "SE1": "10Y1001A1001A44P",
    "SE2": "10Y1001A1001A45N",
    "SE3": "10Y1001A1001A46L",
    "SE4": "10Y1001A1001A47J",
    "DK1": "10YDK-1--------W",
    "DK2": "10YDK-2--------M",
    "FI": "10YFI-1--------U",
    "GB": "10YGB----------A",
    "DE_LU": "10Y1001A1001A82H",
    "NL": "10YNL----------L",
    "LT": "10YLT-1001A0008Q",
    "PL": "10YPL-AREA-----S",
    "EE": "10Y1001A1001A39I"
}

# Connections mellom prisområdene
connections = [
    ("NO1", "NO2"), ("NO1", "NO3"), ("NO1", "NO5"), ("NO2", "NO5"),
    ("NO3", "NO4"), ("NO3", "NO5"), ("NO3", "SE2"), ("NO1", "SE3"),
    ("NO4", "FI"), ("NO2", "DE_LU"), ("NO2", "DK1"), ("NO2", "GB"),
    ("NO2", "NL"), ("NO4", "SE1"), ("NO4", "SE2"), ("SE1", "FI"),
    ("SE1", "SE2"), ("SE2", "SE3"), ("SE3", "DK1"), ("SE3", "FI"),
    ("SE3", "SE4"), ("SE4", "DE_LU"), ("SE4", "DK2"), ("SE4", "LT"),
    ("SE4", "PL"), ("DK1", "DE_LU"), ("DK1", "DK2"), ("DK1", "GB"),
    ("DK1", "NL"), ("DK2", "DE_LU"), ("FI", "EE")
]

# Tidsperiode for datainnhenting
start = pd.Timestamp('20150101', tz='Europe/Brussels')  # Startdato
end = pd.Timestamp('20250101', tz='Europe/Brussels') # Sluttdato

# Maks antall forsøk ved feil
MAX_RETRIES = 3

# Funksjon for å hente data og lagre CSV for én forbindelse
def fetch_and_save(connection):
    from_area, to_area = connection
    retries = 0
    while retries < MAX_RETRIES:
        try:
            # EIC-koder for prisområdene
            from_code = price_areas[from_area]
            to_code = price_areas[to_area]

            # Hent data for flyt fra `from_area` til `to_area`
            print(f"Henter data fra {from_area} til {to_area}...")
            data_from_to = client.query_crossborder_flows(from_code, to_code, start=start, end=end)

            # Hent data for flyt fra `to_area` til `from_area`
            print(f"Henter data fra {to_area} til {from_area}...")
            data_to_from = client.query_crossborder_flows(to_code, from_code, start=start, end=end)

            # Opprett DataFrame for eksport/import
            df_from_to = pd.DataFrame(data_from_to, columns=["Flow"])
            df_from_to['Import'] = 0
            df_from_to['Export'] = df_from_to['Flow']

            df_to_from = pd.DataFrame(data_to_from, columns=["Flow"])
            df_to_from['Import'] = df_to_from['Flow']
            df_to_from['Export'] = 0

            # Kombiner import og eksport
            combined_df = pd.DataFrame(index=df_from_to.index)  # Bruk enhetlig tidsindeks
            combined_df['Timestamp'] = combined_df.index
            combined_df['Import'] = df_to_from['Import']
            combined_df['Export'] = df_from_to['Export']
            combined_df['From'] = from_area
            combined_df['To'] = to_area
            combined_df.reset_index(drop=True, inplace=True)  # Fjern indeksen fra DataFrame

            # Lagre til CSV
            filename = f"Flow_bz_{from_area}_to_{to_area}.csv"
            filepath = os.path.join(os.path.join("..", "results", "Flow_bz_Entso_E"), filename)
            combined_df.to_csv(filepath)
            print(f"Lagret data for {from_area} - {to_area} til {filepath}")

            return  # Hvis vellykket, avslutt løkken
        except (ConnectionError, Exception) as e:
            retries += 1
            print(f"Feil ved henting av data for forbindelsen {from_area} - {to_area}: {e}. Forsøker igjen ({retries}/{MAX_RETRIES})...")
            time.sleep(2 ** retries)  # Eksponentiell backoff

    print(f"Mislyktes med å hente data for forbindelsen {from_area} - {to_area} etter {MAX_RETRIES} forsøk.")

# Bruk ThreadPoolExecutor for parallellisering
# Sørg for at katalogen eksisterer
results_dir = os.path.join("..", "results", "Flow_bz_Entso_E")
os.makedirs(results_dir, exist_ok=True)  # Oppretter katalogen hvis den ikke finnes

with ThreadPoolExecutor(max_workers=3) as executor:  # Juster max_workers etter behov
    executor.map(fetch_and_save, connections)


# **Steg for å lage en samlet CSV-fil med maks flow**
max_flows = []

# Katalog for lagrede CSV-filer
results_dir = os.path.join("..", "results", "Flow_bz_Entso_E")

for connection in connections:
    from_area, to_area = connection
    filename = f"Flow_bz_{from_area}_to_{to_area}.csv"
    filepath = os.path.join(results_dir, filename)

    if os.path.exists(filepath):
        # Les CSV-filen
        df = pd.read_csv(filepath)

        # Finn maksimal flyt
        df['max_flow'] = df[['Import', 'Export']].max(axis=1)  # Finn maks uavhengig av retning
        max_row = df.loc[df['max_flow'].idxmax()]  # Rad med maks flyt
        max_flows.append({
            "Timestamp": max_row['Timestamp'],
            "From": from_area,
            "To": to_area,
            "max_flow": max_row['max_flow']
        })

# Lag en samlet DataFrame og lagre som CSV
if max_flows:
    max_flows_df = pd.DataFrame(max_flows)
    max_flows_df.to_csv(os.path.join(results_dir, "max_flows_summary.csv"), index=False)
    print("Lagret samlet CSV-fil med maksimal flyt for alle forbindelser: max_flows_summary.csv")

# %% Percentil

import os
import pandas as pd
import numpy as np

connections = [
    ("NO1", "NO2"), ("NO1", "NO3"), ("NO1", "NO5"), ("NO2", "NO5"),
    ("NO3", "NO4"), ("NO3", "NO5"), ("NO3", "SE2"), ("NO1", "SE3"),
    ("NO4", "FI"), ("NO2", "DE_LU"), ("NO2", "DK1"), ("NO2", "GB"),
    ("NO2", "NL"), ("NO4", "SE1"), ("NO4", "SE2"), ("SE1", "FI"),
    ("SE1", "SE2"), ("SE2", "SE3"), ("SE3", "DK1"), ("SE3", "FI"),
    ("SE3", "SE4"), ("SE4", "DE_LU"), ("SE4", "DK2"), ("SE4", "LT"),
    ("SE4", "PL"), ("DK1", "DE_LU"), ("DK1", "DK2"), ("DK1", "GB"),
    ("DK1", "NL"), ("DK2", "DE_LU"), ("FI", "EE")
]


# Katalog for lagrede CSV-filer
results_dir = os.path.join("..", "NordicNuclearAnalysis", "results", "Flow_bz_Entso_E")

print(f"Sjekker katalog: {os.path.abspath(results_dir)}")

# Variabel for percentil-verdi (kan enkelt endres her)
PERCENTILE_VALUE = 97.5

# Liste for å samle percentilverdier for hver forbindelse
percentile_flows = []

median_flows = []

for connection in connections:
    from_area, to_area = connection
    filename = f"Flow_bz_{from_area}_to_{to_area}.csv"
    filepath = os.path.join(results_dir, filename)

    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        combined_flows = pd.concat([df['Import'], df['Export']])
        top_20_flows = combined_flows.nlargest(20)
        median_flow = top_20_flows.median()

        median_timestamp = df.loc[top_20_flows.index]['Timestamp'].iloc[0]

        median_flows.append({
            "Timestamp": median_timestamp,
            "From": from_area,
            "To": to_area,
            "median_top_20_flow": median_flow
        })

        # Beregn ønsket percentil
        percentile_value = np.percentile(combined_flows.dropna(), PERCENTILE_VALUE)

        percentile_flows.append({
            "From": from_area,
            "To": to_area,
            f"{PERCENTILE_VALUE}_percentile_flow": percentile_value
        })

# Lag en samlet DataFrame og lagre som CSV
if percentile_flows:
    percentile_flows_df = pd.DataFrame(percentile_flows)
    percentile_flows_df.to_csv(os.path.join(results_dir, f"percentile_{PERCENTILE_VALUE}_flows_summary.csv"), index=False)
    print(f"Lagret CSV-fil med {PERCENTILE_VALUE}-percentilen for alle forbindelser: percentile_{PERCENTILE_VALUE}_flows_summary.csv")



# %%

# median_flows = []
#
# for connection in connections:
#     from_area, to_area = connection
#     filename = f"Flow_bz_{from_area}_to_{to_area}.csv"
#     filepath = os.path.join(results_dir, filename)
#
#     if os.path.exists(filepath):
#         df = pd.read_csv(filepath)
#         combined_flows = pd.concat([df['Import'], df['Export']])
#         top_20_flows = combined_flows.nlargest(20)
#         median_flow = top_20_flows.median()
#
#         median_timestamp = df.loc[top_20_flows.index]['Timestamp'].iloc[0]
#
#         median_flows.append({
#             "Timestamp": median_timestamp,
#             "From": from_area,
#             "To": to_area,
#             "median_top_20_flow": median_flow
#         })
#
# if median_flows:
#     median_flows_df = pd.DataFrame(median_flows)
#     median_flows_df.to_csv(os.path.join(results_dir, "median_top_20_flows_summary.csv"), index=False)
#     print("Lagret CSV-fil med median av topp 20 flyt for alle forbindelser: median_top_20_flows_summary.csv")
#
#
# def summarize_annual_import_export():
#     annual_flows = []
#
#     results_dir = os.path.join("..", "results", "Flow_bz_Entso_E")
#
#     for connection in connections:
#         from_area, to_area = connection
#         filename = f"Flow_bz_{from_area}_to_{to_area}.csv"
#         filepath = os.path.join(results_dir, filename)
#
#         if os.path.exists(filepath):
#             # Les CSV-filen
#             df = pd.read_csv(filepath)
#
#             # Konverter 'Timestamp' til datetime med eksplisitt UTC
#             df['Timestamp'] = pd.to_datetime(df['Timestamp'], utc=True, errors='coerce')
#
#             if df['Timestamp'].isna().all():
#                 print(f"Advarsel: Alle tidsstempler er ugyldige for {filename}. Skipping.")
#                 continue
#
#             # Ekstraher år fra tidsstemplene
#             df['Year'] = df['Timestamp'].dt.year
#
#             # Gruppér etter år og summer import og eksport
#             yearly_summary = df.groupby('Year')[['Import', 'Export']].sum().reset_index()
#             yearly_summary['From'] = from_area
#             yearly_summary['To'] = to_area
#
#             # Legg til resultatene i listen
#             annual_flows.extend(yearly_summary.to_dict('records'))
#
#     if annual_flows:
#         annual_flows_df = pd.DataFrame(annual_flows)
#         annual_flows_df = annual_flows_df[['Year', 'From', 'To', 'Import', 'Export']]  # Rekkefølge på kolonner
#         output_file = os.path.join(results_dir, "annual_imp_exp_BZ_summary.csv")
#         annual_flows_df.to_csv(output_file, index=False)
#         print(f"CSV-fil med årlig import og eksport mellom alle prisområder: {output_file}")
#
# summarize_annual_import_export()