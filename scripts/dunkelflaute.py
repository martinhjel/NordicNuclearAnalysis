
'''
Description:
This script allows the user to select regions, technologies, and aggregation periods (e.g., 1–14 days)
to identify the three lowest average renewable production periods for each interval. Results are saved
to an Excel file named accordingly.
'''

import pandas as pd

# === INPUT ===
file_path = r"C:\Users\einar\NTNU\MSc. - Kjernekraft og PowerGAMA - General\TET4900\Data\timeseries_profiles_stock.csv"
region = ["DK1", "DK2", "SE4", "DE", "NL", "PL"]
tech = ["windoff", "windon", "solar"]
aggregation_days_list = list(range(1, 15))  # eller f.eks. [3, 4, 5]

df = pd.read_csv(file_path)
df["time"] = pd.to_datetime(df["time"], utc=True)
df.set_index("time", inplace=True)

def extract_columns(df, regions, techs):
    return [
        col for col in df.columns
        if any(col.startswith(t + "_") and col.split("_")[-1] in regions for t in techs)
    ]

selected_cols = extract_columns(df, region, tech)
df_filtered = df[selected_cols].copy()

results = []
for days in aggregation_days_list:
    rule = f"{days}D"
    df_agg = df_filtered.resample(rule).mean()
    df_mean = df_agg.mean(axis=1).sort_values().head(3)  # Tre laveste perioder

    for timestamp, value in df_mean.items():
        results.append({
            "aggregation_days": days,
            "start_date": timestamp,
            "mean_value": value
        })

result_df = pd.DataFrame(results)
result_df["start_date"] = result_df["start_date"].dt.tz_localize(None)

region_label = "_".join(region)
tech_label = "_".join(tech)
filename = f"low_inflow_periods_{region_label}_{tech_label}.xlsx"
result_df.to_excel(filename, index=False)

print(f"✅ Lagret til: {filename}")
